"""Tests for GPP utility scoring."""

from __future__ import annotations

import json
from pathlib import Path

from megax.crowd import build_crowd_matrix
from megax.ev import compute_ev
from megax.probability import build_score_matrix_from_match
from megax.tipsport.offer import parse_match
from megax.utility import (
    MAX_GPP_CROWD_SHARE,
    MIN_GPP_CROWD_SHARE,
    compute_match_analysis,
    gpp_alpha_from_field_size,
    resolve_gpp_alpha,
    utility_score,
)

FIXTURE = Path(__file__).parent / "fixtures" / "comp120_match_allevents.json"


def _prob_and_crowd():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    match = parse_match(raw["matches"][0])
    assert match is not None
    prob = build_score_matrix_from_match(match)
    assert prob is not None
    money = {
        "tipsport": {"home": 60.0, "draw": 20.0, "away": 20.0, "over": 50.0, "under": 50.0},
        "fortuna": {"home": 80.0, "draw": 10.0, "away": 10.0, "over": 55.0, "under": 45.0},
        "sazkabet": {"home": 70.0, "draw": 15.0, "away": 15.0, "over": 52.0, "under": 48.0},
    }
    crowd = build_crowd_matrix(prob, money)
    return prob, crowd


def test_utility_prefers_low_crowd_at_same_ev() -> None:
    high = utility_score(3.0, 0.20, alpha=1.1)
    low = utility_score(3.0, 0.05, alpha=1.1)
    assert low > high


def test_gpp_alpha_grows_with_field_size() -> None:
    small = gpp_alpha_from_field_size(5_000)
    large = gpp_alpha_from_field_size(50_000)
    assert large > small


def test_resolve_gpp_alpha_honors_override() -> None:
    assert resolve_gpp_alpha(50_000, override=1.35) == 1.35


def test_compute_match_analysis_returns_ev_and_gpp_top3() -> None:
    prob, crowd = _prob_and_crowd()
    analysis = compute_match_analysis(prob, crowd, field_size=50_000, top_n=3)
    assert len(analysis.ev.top) == 3
    assert len(analysis.gpp_top) == 3
    assert analysis.ev.best.ev >= analysis.ev.top[1].ev
    assert analysis.gpp_best.utility >= analysis.gpp_top[1].utility


def test_plzen_gpp_favors_reasonable_scores() -> None:
    from test_probability import _plzen_match

    prob = build_score_matrix_from_match(_plzen_match())
    assert prob is not None
    money = {
        "tipsport": {"home": 62, "draw": 18, "away": 20, "over": 48, "under": 52},
        "fortuna": {"home": 87, "draw": 6, "away": 7, "over": 55, "under": 45},
        "sazkabet": {"home": 85, "draw": 6, "away": 9, "over": 52, "under": 48},
    }
    crowd = build_crowd_matrix(prob, money)
    analysis = compute_match_analysis(prob, crowd, field_size=50_000)
    best = analysis.gpp_best
    assert best.home <= 4 and best.away <= 4
    assert MIN_GPP_CROWD_SHARE <= best.crowd_share <= MAX_GPP_CROWD_SHARE
    assert best.ev >= analysis.ev.best.ev * 0.85 - 1e-9
