"""Tests for expected points (EV) on exact-score tips."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from megax.ev import (
    TipCandidate,
    compute_ev,
    expected_points,
    format_tip,
    iter_tip_candidates,
    parse_tip,
    rank_tips_by_ev,
)
from megax.probability import build_score_matrix_from_match, probability
from megax.scoring import points
from megax.tipsport.offer import parse_match

FIXTURE = Path(__file__).parent / "fixtures" / "comp120_match_allevents.json"


def _prob():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    match = parse_match(raw["matches"][0])
    assert match is not None
    result = build_score_matrix_from_match(match)
    assert result is not None
    return result


def test_format_and_parse_tip() -> None:
    assert format_tip(2, 1) == "2:1"
    assert parse_tip("2:1") == (2, 1)
    assert parse_tip("bad") is None


def test_expected_points_matches_manual_sum() -> None:
    prob = _prob()
    tip_home, tip_away = 2, 1
    manual = 0.0
    for actual_home in range(prob.grid_size):
        for actual_away in range(prob.grid_size):
            manual += prob.matrix[actual_home][actual_away] * points(
                tip_home,
                tip_away,
                actual_home,
                actual_away,
            )
    assert expected_points(prob, tip_home, tip_away) == pytest.approx(manual)


def test_exact_score_tip_on_point_mass() -> None:
    grid = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    assert expected_points(grid, 2, 1) == pytest.approx(10.0)
    assert expected_points(grid, 2, 0) == pytest.approx(4.0)


def test_ev_within_scoring_bounds() -> None:
    prob = _prob()
    for candidate in iter_tip_candidates(prob):
        assert 0.0 <= candidate.ev <= 10.0


def test_rank_tips_by_ev_returns_top_three_descending() -> None:
    prob = _prob()
    top = rank_tips_by_ev(prob, top_n=3)
    assert len(top) == 3
    assert top[0].ev >= top[1].ev >= top[2].ev
    assert all(isinstance(candidate, TipCandidate) for candidate in top)


def test_compute_ev_exposes_best_and_top3() -> None:
    prob = _prob()
    result = compute_ev(prob, top_n=3)
    assert result.best == result.top[0]
    assert result.top3[0].ev >= result.top3[1].ev
    assert result.grid_size == prob.grid_size


def test_plzen_best_ev_is_sensible_home_chalk() -> None:
    from test_probability import _plzen_match

    prob = build_score_matrix_from_match(_plzen_match())
    assert prob is not None
    result = compute_ev(prob, top_n=3)
    best = result.best
    assert best.home > best.away
    assert best.home <= 3
    assert best.ev > 3.0
    assert probability(prob, best.home, best.away) > 0.03
