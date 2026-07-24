"""Score probability model v3 — 1X2 + team O/U + match totals + low-score fix."""

from __future__ import annotations

from dataclasses import dataclass

from megax.low_score import DEFAULT_LOW_SCORE_RHO
from megax.market_math import devig_two_way
from megax.match_total_mu import (
    DEFAULT_TOTAL_BLEND_WEIGHT,
    MatchTotalEstimate,
    blend_team_mus_with_match_total,
    estimate_match_total_mu,
)
from megax.score_prior import (
    DEFAULT_GRID_SIZE,
    ScorePriorResult,
    devig_1x2_probs,
    fit_market_score_prior,
)
from megax.team_mu import TeamMuEstimate, estimate_team_mus
from megax.tipsport.offer import MegaxMatch, MatchOdds as TipsportMatchOdds

MODEL_VERSION = "megax_probability_v3"


@dataclass(frozen=True)
class ScoreMatrixResult:
    model_version: str
    matrix: tuple[tuple[float, ...], ...]
    grid_size: int
    p_home: float
    p_draw: float
    p_away: float
    home_mu: float
    away_mu: float
    expected_total: float
    goal_diff_away_home: float
    p_over_2_5: float | None
    team_estimate: TeamMuEstimate
    prior: ScorePriorResult
    team_total_mu: float
    match_total_estimate: MatchTotalEstimate | None
    total_blend_weight: float
    low_score_rho: float


def build_score_matrix_from_match(
    match: MegaxMatch,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    total_blend_weight: float = DEFAULT_TOTAL_BLEND_WEIGHT,
    low_score_rho: float = DEFAULT_LOW_SCORE_RHO,
) -> ScoreMatrixResult | None:
    """Build P(home, away) from 1X2, team O/U, match totals, and max-entropy fit."""
    odds = match.odds
    probs_1x2 = devig_1x2_probs(odds.home, odds.draw, odds.away)
    if probs_1x2 is None:
        return None
    p_home, p_draw, p_away = probs_1x2

    team_estimate = estimate_team_mus(odds.home_team_lines, odds.away_team_lines)
    if team_estimate is None:
        return None

    team_total_mu = team_estimate.home_mu + team_estimate.away_mu
    home_mu = team_estimate.home_mu
    away_mu = team_estimate.away_mu
    blend_weight = 0.0

    match_total_estimate = estimate_match_total_mu(odds.match_total_lines)
    if match_total_estimate is not None:
        home_mu, away_mu = blend_team_mus_with_match_total(
            team_estimate,
            match_total_estimate,
            total_weight=total_blend_weight,
        )
        blend_weight = min(max(total_blend_weight, 0.0), 1.0)

    expected_total = home_mu + away_mu
    goal_diff = away_mu - home_mu

    prior = fit_market_score_prior(
        expected_total=expected_total,
        goal_diff_away_home=goal_diff,
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        grid_size=grid_size,
        low_score_rho=low_score_rho,
    )
    if prior is None:
        return None

    p_over_2_5 = None
    if odds.over_2_5 is not None and odds.under_2_5 is not None:
        fair_ou = devig_two_way(odds.over_2_5, odds.under_2_5)
        if fair_ou is not None:
            p_over_2_5 = fair_ou[0]

    return ScoreMatrixResult(
        model_version=MODEL_VERSION,
        matrix=prior.matrix,
        grid_size=grid_size,
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        home_mu=home_mu,
        away_mu=away_mu,
        expected_total=expected_total,
        goal_diff_away_home=goal_diff,
        p_over_2_5=p_over_2_5,
        team_estimate=team_estimate,
        prior=prior,
        team_total_mu=team_total_mu,
        match_total_estimate=match_total_estimate,
        total_blend_weight=blend_weight,
        low_score_rho=low_score_rho,
    )


def build_score_matrix_from_tipsport(
    odds: TipsportMatchOdds,
    *,
    home: str = "",
    away: str = "",
    match_id: int = 0,
    grid_size: int = DEFAULT_GRID_SIZE,
    total_blend_weight: float = DEFAULT_TOTAL_BLEND_WEIGHT,
    low_score_rho: float = DEFAULT_LOW_SCORE_RHO,
) -> ScoreMatrixResult | None:
    """Convenience wrapper when only MatchOdds is available."""
    from datetime import datetime, timezone

    from megax.tipsport.offer import MegaxMatch

    stub = MegaxMatch(
        match_id=match_id,
        name=f"{home} - {away}".strip(" -"),
        home=home,
        away=away,
        kickoff_at=datetime.now(timezone.utc),
        odds=odds,
        match_type="PREMATCH",
        ended=False,
        competition_id=120,
    )
    return build_score_matrix_from_match(
        stub,
        grid_size=grid_size,
        total_blend_weight=total_blend_weight,
        low_score_rho=low_score_rho,
    )


def probability(matrix: ScoreMatrixResult | tuple[tuple[float, ...], ...], home: int, away: int) -> float:
    grid = matrix.matrix if isinstance(matrix, ScoreMatrixResult) else matrix
    if home < 0 or away < 0:
        return 0.0
    if home >= len(grid) or away >= len(grid[0]):
        return 0.0
    return grid[home][away]
