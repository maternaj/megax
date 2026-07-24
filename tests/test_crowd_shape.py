"""Tests for crowd behavioral shape (γ, δ, Prelec)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from megax.crowd import build_crowd_matrix
from megax.crowd_shape import (
    DEFAULT_CROWD_PRELEC_ALPHA,
    DEFAULT_CROWD_TAIL_GAMMA,
    DEFAULT_CROWD_ZERO_ZERO_DELTA,
    DEFAULT_CROWD_ZERO_ZERO_MIN,
    apply_prelec,
    apply_tail_dampening,
    apply_zero_zero_aversion,
    apply_zero_zero_floor,
    prelec_weight,
    shape_crowd_matrix,
)
from megax.probability import build_score_matrix_from_match
from megax.tipsport.offer import parse_match

FIXTURE = Path(__file__).parent / "fixtures" / "comp120_match_allevents.json"
sys.path.insert(0, str(Path(__file__).parent))
from test_probability import _plzen_match  # noqa: E402


def _prob():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    match = parse_match(raw["matches"][0])
    assert match is not None
    result = build_score_matrix_from_match(match)
    assert result is not None
    return result


def _tail_mass(matrix: tuple[tuple[float, ...], ...]) -> float:
    total = 0.0
    for home, row in enumerate(matrix):
        for away, value in enumerate(row):
            if home >= 3 or away >= 3:
                total += value
    return total


def test_tail_dampening_reduces_high_scoring_mass() -> None:
    prob = _prob()
    damped = apply_tail_dampening(prob.matrix, gamma=DEFAULT_CROWD_TAIL_GAMMA)
    assert _tail_mass(damped) < _tail_mass(prob.matrix)
    assert abs(sum(sum(row) for row in damped) - 1.0) < 1e-6


def test_zero_zero_aversion_shifts_mass_to_chalk_draws() -> None:
    prob = _prob()
    adjusted = apply_zero_zero_aversion(prob.matrix, delta=DEFAULT_CROWD_ZERO_ZERO_DELTA)
    assert adjusted[0][0] < prob.matrix[0][0]
    assert adjusted[1][1] > prob.matrix[1][1]
    assert adjusted[1][0] > prob.matrix[1][0]
    assert abs(sum(sum(row) for row in adjusted) - 1.0) < 1e-6


def test_prelec_skips_zero_zero_cell() -> None:
    prob = _prob()
    base = apply_zero_zero_aversion(prob.matrix)
    skipped = apply_prelec(base, skip_cells=frozenset({(0, 0)}))
    assert skipped[0][0] == base[0][0]


def test_zero_zero_floor_enforces_minimum_share() -> None:
    tiny = (
        (0.001, 0.499),
        (0.499, 0.001),
    )
    floored = apply_zero_zero_floor(tiny, min_share=0.015)
    assert floored[0][0] >= 0.015
    assert abs(sum(sum(row) for row in floored) - 1.0) < 1e-6


def test_prelec_boosts_likely_cells_relative_to_tail() -> None:
    assert prelec_weight(0.20, alpha=DEFAULT_CROWD_PRELEC_ALPHA) / 0.20 > (
        prelec_weight(0.01, alpha=DEFAULT_CROWD_PRELEC_ALPHA) / 0.01
    )


def test_shaped_crowd_is_chalkier_than_objective_p() -> None:
    prob = _prob()
    money = {
        "tipsport": {"home": 60.0, "draw": 20.0, "away": 20.0, "over": 50.0, "under": 50.0},
        "fortuna": {"home": 80.0, "draw": 10.0, "away": 10.0, "over": 55.0, "under": 45.0},
        "sazkabet": {"home": 70.0, "draw": 15.0, "away": 15.0, "over": 52.0, "under": 48.0},
    }
    crowd = build_crowd_matrix(prob, money)

    assert crowd.matrix[0][0] < prob.matrix[0][0]
    assert crowd.matrix[0][0] >= DEFAULT_CROWD_ZERO_ZERO_MIN - 1e-9
    assert _tail_mass(crowd.matrix) < _tail_mass(prob.matrix)
    assert crowd.prelec_alpha == DEFAULT_CROWD_PRELEC_ALPHA
    assert crowd.tail_gamma == DEFAULT_CROWD_TAIL_GAMMA
    assert crowd.zero_zero_delta == DEFAULT_CROWD_ZERO_ZERO_DELTA


def test_plzen_tuning_is_less_extreme_than_before() -> None:
    prob = build_score_matrix_from_match(_plzen_match())
    assert prob is not None
    money = {
        "tipsport": {"home": 62, "draw": 18, "away": 20, "over": 48, "under": 52},
        "fortuna": {"home": 87, "draw": 6, "away": 7, "over": 55, "under": 45},
        "sazkabet": {"home": 85, "draw": 6, "away": 9, "over": 52, "under": 48},
    }
    crowd = build_crowd_matrix(prob, money)

    ratio_20_30 = crowd.matrix[2][0] / crowd.matrix[3][0]
    top3 = crowd.matrix[1][0] + crowd.matrix[2][0] + crowd.matrix[2][1]

    assert crowd.matrix[0][0] >= DEFAULT_CROWD_ZERO_ZERO_MIN
    assert crowd.matrix[3][0] >= 0.015
    assert ratio_20_30 < 20
    assert top3 < 0.80
