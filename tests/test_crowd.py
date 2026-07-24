"""Tests for crowd pick model."""

from __future__ import annotations

import json
from pathlib import Path

from megax.crowd import (
    DEFAULT_CROWD_BLEND_TO_P,
    blend_outcome_mass,
    build_crowd_matrix,
    outcome_mass_from_money,
)
from megax.probability import build_score_matrix_from_match
from megax.tipsport.offer import parse_match

FIXTURE = Path(__file__).parent / "fixtures" / "comp120_match_allevents.json"


def _prob():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    match = parse_match(raw["matches"][0])
    assert match is not None
    result = build_score_matrix_from_match(match)
    assert result is not None
    return result


def test_crowd_uses_soft_money_for_outcome_mass() -> None:
    prob = _prob()
    money = {
        "tipsport": {"home": 60.0, "draw": 20.0, "away": 20.0, "over": 50.0, "under": 50.0},
        "fortuna": {"home": 80.0, "draw": 10.0, "away": 10.0, "over": 55.0, "under": 45.0},
        "sazkabet": {"home": 70.0, "draw": 15.0, "away": 15.0, "over": 52.0, "under": 48.0},
    }
    mass, source = outcome_mass_from_money(money)
    assert source == "soft_money"
    assert mass is not None
    assert mass[0] > mass[2]

    crowd = build_crowd_matrix(prob, money)
    assert abs(sum(sum(row) for row in crowd.matrix) - 1.0) < 1e-6
    assert crowd.source == "soft_money"
    assert crowd.outcome_mass_raw is not None
    assert crowd.blend_to_p == DEFAULT_CROWD_BLEND_TO_P
    assert crowd.outcome_mass[0] < crowd.outcome_mass_raw[0]


def test_blend_outcome_mass_pulls_toward_p() -> None:
    prob = _prob()
    money_mass = (0.80, 0.10, 0.10)
    blended = blend_outcome_mass(money_mass, prob, blend_to_p=0.30)
    assert blended[0] < money_mass[0]
    assert blended[0] > prob.p_home
    assert abs(sum(blended) - 1.0) < 1e-6

    no_blend = blend_outcome_mass(money_mass, prob, blend_to_p=0.0)
    assert no_blend == money_mass

    full_blend = blend_outcome_mass(money_mass, prob, blend_to_p=1.0)
    assert abs(full_blend[0] - prob.p_home) < 1e-6
    assert abs(full_blend[1] - prob.p_draw) < 1e-6
    assert abs(full_blend[2] - prob.p_away) < 1e-6


def test_crowd_fallback_without_money() -> None:
    prob = _prob()
    money = {
        "tipsport": {"home": None, "draw": None, "away": None, "over": None, "under": None},
        "fortuna": {"home": None, "draw": None, "away": None, "over": None, "under": None},
        "sazkabet": {"home": None, "draw": None, "away": None, "over": None, "under": None},
    }
    crowd = build_crowd_matrix(prob, money)
    assert crowd.source == "fallback_odds"
    assert crowd.outcome_mass == (prob.p_home, prob.p_draw, prob.p_away)
    assert crowd.outcome_mass_raw is None
    assert crowd.blend_to_p == 1.0
