"""Estimate team goal rates from participant Over/Under lines."""

from __future__ import annotations

from dataclasses import dataclass

from megax.market_math import devig_two_way, p_over_poisson


@dataclass(frozen=True)
class TeamOuLine:
    line: float
    over: float
    under: float


@dataclass(frozen=True)
class TeamMuEstimate:
    home_mu: float
    away_mu: float
    home_lines_used: int
    away_lines_used: int
    source: str


def invert_team_mu(line: TeamOuLine, *, mu_max: float = 5.5) -> float | None:
    fair = devig_two_way(line.over, line.under)
    if fair is None:
        return None
    target_over = fair[0]

    lo, hi = 0.05, mu_max
    p_lo = p_over_team_goals(line.line, lo)
    p_hi = p_over_team_goals(line.line, hi)
    if p_lo is None or p_hi is None:
        return None
    if target_over < min(p_lo, p_hi) - 1e-9 or target_over > max(p_lo, p_hi) + 1e-9:
        return None

    for _ in range(56):
        mid = (lo + hi) / 2.0
        p_mid = p_over_team_goals(line.line, mid)
        if p_mid is None:
            return None
        if p_mid < target_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def p_over_team_goals(line: float, mu: float) -> float | None:
    return p_over_poisson(line, mu)


def _blend_team_mu(lines: tuple[TeamOuLine, ...]) -> tuple[float | None, int]:
    estimates: list[float] = []
    for line in lines:
        mu = invert_team_mu(line)
        if mu is not None:
            estimates.append(mu)
    if not estimates:
        return None, 0
    return sum(estimates) / len(estimates), len(estimates)


def estimate_team_mus(
    home_lines: tuple[TeamOuLine, ...],
    away_lines: tuple[TeamOuLine, ...],
) -> TeamMuEstimate | None:
    home_mu, home_n = _blend_team_mu(home_lines)
    away_mu, away_n = _blend_team_mu(away_lines)
    if home_mu is None or away_mu is None:
        return None
    return TeamMuEstimate(
        home_mu=home_mu,
        away_mu=away_mu,
        home_lines_used=home_n,
        away_lines_used=away_n,
        source="team_participant_ou",
    )
