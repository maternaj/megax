"""GPP utility — EV vs crowd ownership for exact-score tips."""

from __future__ import annotations

import math
from dataclasses import dataclass

from megax.crowd import CrowdMatrixResult
from megax.ev import EvResult, compute_ev, format_tip, iter_tip_candidates
from megax.probability import ScoreMatrixResult

MIN_CROWD_SHARE = 1e-8
MIN_GPP_CROWD_SHARE = 0.005
MAX_GPP_CROWD_SHARE = 0.12
DEFAULT_GPP_EV_RATIO = 0.85
DEFAULT_GPP_ALPHA = 1.10
DEFAULT_FIELD_SIZE = 50_000
DEFAULT_GPP_MAX_GOALS = 4


@dataclass(frozen=True)
class UtilityCandidate:
    home: int
    away: int
    ev: float
    crowd_share: float
    utility: float

    @property
    def score(self) -> str:
        return format_tip(self.home, self.away)


@dataclass(frozen=True)
class MatchTipAnalysis:
    ev: EvResult
    gpp_alpha: float
    gpp_best: UtilityCandidate
    gpp_top: tuple[UtilityCandidate, ...]

    @property
    def gpp_top3(self) -> tuple[UtilityCandidate, UtilityCandidate, UtilityCandidate]:
        if len(self.gpp_top) < 3:
            raise ValueError("GPP result has fewer than 3 candidates")
        return self.gpp_top[0], self.gpp_top[1], self.gpp_top[2]


def crowd_share(
    crowd: CrowdMatrixResult | tuple[tuple[float, ...], ...],
    home: int,
    away: int,
) -> float:
    grid = crowd.matrix if isinstance(crowd, CrowdMatrixResult) else crowd
    if home < 0 or away < 0 or home >= len(grid) or away >= len(grid[0]):
        return 0.0
    return max(grid[home][away], 0.0)


def gpp_alpha_from_field_size(field_size: int) -> float:
    """Heuristic α(N): larger fields favour more contrarian picks."""
    if field_size <= 0:
        return DEFAULT_GPP_ALPHA
    return 1.0 + 0.12 * math.log10(max(field_size, 1_000) / 10_000)


def resolve_gpp_alpha(field_size: int, *, override: float | None = None) -> float:
    if override is not None:
        return max(override, 0.0)
    return gpp_alpha_from_field_size(field_size)


def utility_score(ev: float, crowd: float, *, alpha: float) -> float:
    """U(T) = EV(T) / C(T)^α."""
    if alpha <= 0:
        return ev
    safe_crowd = max(crowd, MIN_CROWD_SHARE)
    return ev / (safe_crowd**alpha)


def iter_utility_candidates(
    prob: ScoreMatrixResult,
    crowd: CrowdMatrixResult,
    *,
    alpha: float,
    max_goals: int | None = DEFAULT_GPP_MAX_GOALS,
    min_ev: float = 0.0,
) -> tuple[UtilityCandidate, ...]:
    candidates: list[UtilityCandidate] = []
    for tip in iter_tip_candidates(prob, max_goals=max_goals):
        if tip.ev < min_ev:
            continue
        share = crowd_share(crowd, tip.home, tip.away)
        if share < MIN_GPP_CROWD_SHARE or share > MAX_GPP_CROWD_SHARE:
            continue
        candidates.append(
            UtilityCandidate(
                home=tip.home,
                away=tip.away,
                ev=tip.ev,
                crowd_share=share,
                utility=utility_score(tip.ev, share, alpha=alpha),
            )
        )
    return tuple(candidates)


def rank_tips_by_utility(
    prob: ScoreMatrixResult,
    crowd: CrowdMatrixResult,
    *,
    alpha: float,
    top_n: int = 3,
    max_goals: int | None = DEFAULT_GPP_MAX_GOALS,
    min_ev: float = 0.0,
) -> tuple[UtilityCandidate, ...]:
    ranked = sorted(
        iter_utility_candidates(
            prob,
            crowd,
            alpha=alpha,
            max_goals=max_goals,
            min_ev=min_ev,
        ),
        key=lambda candidate: (-candidate.utility, -candidate.ev, -candidate.home, -candidate.away),
    )
    if top_n <= 0:
        return ()
    return tuple(ranked[:top_n])


def compute_match_analysis(
    prob: ScoreMatrixResult,
    crowd: CrowdMatrixResult,
    *,
    field_size: int = DEFAULT_FIELD_SIZE,
    gpp_alpha: float | None = None,
    gpp_ev_ratio: float | None = None,
    alpha_boost: float = 0.0,
    top_n: int = 3,
) -> MatchTipAnalysis:
    alpha = resolve_gpp_alpha(field_size, override=gpp_alpha) + max(alpha_boost, 0.0)
    ev = compute_ev(prob, top_n=top_n)
    ratio = DEFAULT_GPP_EV_RATIO if gpp_ev_ratio is None else gpp_ev_ratio
    min_ev = ev.best.ev * ratio
    gpp_top = rank_tips_by_utility(
        prob,
        crowd,
        alpha=alpha,
        top_n=max(top_n, 1),
        min_ev=min_ev,
    )
    if not gpp_top:
        raise ValueError("No utility candidates")
    return MatchTipAnalysis(
        ev=ev,
        gpp_alpha=alpha,
        gpp_best=gpp_top[0],
        gpp_top=gpp_top,
    )
