"""Estimate match total goals from Asian total O/U lines."""

from __future__ import annotations

from dataclasses import dataclass

from megax.market_math import devig_two_way, p_over_poisson
from megax.team_mu import TeamMuEstimate, TeamOuLine

DEFAULT_TOTAL_BLEND_WEIGHT = 0.5


@dataclass(frozen=True)
class MatchTotalEstimate:
    total_mu: float
    lines_used: int
    source: str


def invert_match_total_mu(line: TeamOuLine, *, mu_max: float = 8.0) -> float | None:
    """Invert a match-total Asian line into Poisson total-goals μ."""
    fair = devig_two_way(line.over, line.under)
    if fair is None:
        return None
    target_over = fair[0]

    lo, hi = 0.05, mu_max
    p_lo = p_over_poisson(line.line, lo)
    p_hi = p_over_poisson(line.line, hi)
    if p_lo is None or p_hi is None:
        return None
    if target_over < min(p_lo, p_hi) - 1e-9 or target_over > max(p_lo, p_hi) + 1e-9:
        return None

    for _ in range(56):
        mid = (lo + hi) / 2.0
        p_mid = p_over_poisson(line.line, mid)
        if p_mid is None:
            return None
        if p_mid < target_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _line_balance_weight(line: TeamOuLine) -> float:
    """Weight lines by balance — odds near 50/50 are sharper anchors."""
    fair = devig_two_way(line.over, line.under)
    if fair is None:
        return 0.0
    imbalance = abs(fair[0] - 0.5)
    return max(0.05, 1.0 - 2.0 * imbalance)


def estimate_match_total_mu(lines: tuple[TeamOuLine, ...]) -> MatchTotalEstimate | None:
    weighted: list[tuple[float, float]] = []
    for line in lines:
        mu = invert_match_total_mu(line)
        if mu is None:
            continue
        weight = _line_balance_weight(line)
        if weight <= 0:
            continue
        weighted.append((mu, weight))
    if not weighted:
        return None
    total_weight = sum(weight for _, weight in weighted)
    blended = sum(mu * weight for mu, weight in weighted) / total_weight
    return MatchTotalEstimate(
        total_mu=blended,
        lines_used=len(weighted),
        source="match_asian_total",
    )


def blend_team_mus_with_match_total(
    team_estimate: TeamMuEstimate,
    match_total: MatchTotalEstimate,
    *,
    total_weight: float = DEFAULT_TOTAL_BLEND_WEIGHT,
) -> tuple[float, float]:
    """Scale team λ while preserving home/away ratio to match blended total μ."""
    team_total = team_estimate.home_mu + team_estimate.away_mu
    if team_total <= 0:
        return team_estimate.home_mu, team_estimate.away_mu
    weight = min(max(total_weight, 0.0), 1.0)
    blended_total = (1.0 - weight) * team_total + weight * match_total.total_mu
    scale = blended_total / team_total
    return team_estimate.home_mu * scale, team_estimate.away_mu * scale
