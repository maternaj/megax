"""Tests for score probability model v3."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from megax.market_math import p_over_poisson
from megax.match_total_mu import estimate_match_total_mu
from megax.probability import MODEL_VERSION, build_score_matrix_from_match, probability
from megax.score_prior import matrix_moments
from megax.team_mu import TeamOuLine, estimate_team_mus, invert_team_mu
from megax.tipsport.offer import MatchOdds, MegaxMatch, parse_match

FIXTURE = Path(__file__).parent / "fixtures" / "comp120_match_allevents.json"

PLZEN_ODDS = {
    "home": 1.54,
    "draw": 4.46,
    "away": 5.86,
    "over_2_5": 1.70,
    "under_2_5": 2.19,
    "home_team_lines": (
        TeamOuLine(0.5, 1.09, 6.12),
        TeamOuLine(1.5, 1.53, 2.33),
        TeamOuLine(2.0, 2.05, 1.69),
    ),
    "away_team_lines": (
        TeamOuLine(0.5, 1.54, 2.32),
        TeamOuLine(1.0, 2.47, 1.47),
    ),
    "match_total_lines": (
        TeamOuLine(2.5, 1.70, 2.19),
        TeamOuLine(3.0, 2.16, 1.71),
        TeamOuLine(2.0, 1.31, 3.43),
    ),
}


def _plzen_match() -> MegaxMatch:
    return MegaxMatch(
        match_id=8212280,
        name="Viktoria Plzeň - Slovan Liberec",
        home="Viktoria Plzeň",
        away="Slovan Liberec",
        kickoff_at=datetime(2026, 7, 25, 15, 0, tzinfo=timezone.utc),
        odds=MatchOdds(**PLZEN_ODDS),
        match_type="PREMATCH",
        ended=False,
        competition_id=120,
    )


def test_model_version_is_v3() -> None:
    assert MODEL_VERSION == "megax_probability_v3"


def test_expected_total_from_team_lines() -> None:
    estimate = estimate_team_mus(PLZEN_ODDS["home_team_lines"], PLZEN_ODDS["away_team_lines"])
    assert estimate is not None
    assert 1.8 < estimate.home_mu < 2.6
    assert 0.8 < estimate.away_mu < 1.4


def test_build_score_matrix_from_fixture_match() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    match = parse_match(raw["matches"][0])
    assert match is not None
    assert match.odds.home_team_lines
    assert match.odds.away_team_lines
    assert len(match.odds.match_total_lines) >= 3
    result = build_score_matrix_from_match(match)
    assert result is not None
    assert result.model_version == MODEL_VERSION
    assert result.match_total_estimate is not None
    assert abs(sum(sum(row) for row in result.matrix) - 1.0) < 1e-6


def test_plzen_favorite_prefers_low_draw_over_big_home_win() -> None:
    result = build_score_matrix_from_match(_plzen_match())
    assert result is not None
    p00 = probability(result, 0, 0)
    p40 = probability(result, 4, 0)
    assert p00 > p40
    assert p00 < 0.07
    assert p40 < 0.05
    away_clean_sheet = sum(probability(result, i, 0) for i in range(6))
    assert away_clean_sheet < 0.40


def test_match_total_blend_moves_total_toward_market() -> None:
    match = _plzen_match()
    with_blend = build_score_matrix_from_match(match)
    without_blend = build_score_matrix_from_match(match, total_blend_weight=0.0)
    assert with_blend is not None
    assert without_blend is not None
    assert with_blend.match_total_estimate is not None
    market_total = with_blend.match_total_estimate.total_mu
    assert abs(with_blend.expected_total - market_total) < abs(
        without_blend.expected_total - market_total
    )
    assert with_blend.team_total_mu == without_blend.expected_total


def test_low_score_rho_increases_drawish_low_scores() -> None:
    match = _plzen_match()
    with_rho = build_score_matrix_from_match(match, low_score_rho=-0.10)
    without_rho = build_score_matrix_from_match(match, low_score_rho=0.0)
    assert with_rho is not None
    assert without_rho is not None
    assert probability(with_rho, 0, 0) > probability(without_rho, 0, 0)


def test_matrix_still_matches_1x2_marginals() -> None:
    result = build_score_matrix_from_match(_plzen_match())
    assert result is not None
    _, _, fit_home, fit_draw, fit_away = matrix_moments(result.matrix)
    assert abs(fit_home - result.p_home) < 1e-4
    assert abs(fit_draw - result.p_draw) < 1e-4
    assert abs(fit_away - result.p_away) < 1e-4


def test_p_over_poisson_half_and_integer_lines() -> None:
    assert invert_team_mu(TeamOuLine(0.5, 1.54, 2.32)) is not None
    assert invert_team_mu(TeamOuLine(1.0, 2.47, 1.47)) is not None
    assert p_over_poisson(2.0, 2.5) is not None
    assert estimate_match_total_mu(PLZEN_ODDS["match_total_lines"]) is not None
