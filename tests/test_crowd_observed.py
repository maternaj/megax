"""Tests for sparse direct crowd C matrix."""

from __future__ import annotations

import json
from pathlib import Path

from megax.crowd_observed import (
    LONGSHOT_ODDS_THRESHOLD,
    build_crowd_matrix_from_cells,
    crowd_score_constraint,
    expected_odds_from_prob,
    merge_api_top3_into_cells,
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


def test_longshot_interval_when_odds_above_100() -> None:
    prob = _prob()
    longshot = None
    for i in range(prob.grid_size):
        for j in range(prob.grid_size):
            if expected_odds_from_prob(prob, i, j) > LONGSHOT_ODDS_THRESHOLD:
                longshot = (i, j)
                break
        if longshot:
            break
    assert longshot is not None
    c = crowd_score_constraint(prob, longshot[0], longshot[1], pct=1.0)
    assert c.is_longshot_floor
    assert c.lo_pct == 0.0
    assert c.hi_pct == 0.5


def test_regular_interval_is_plus_minus_half_percent() -> None:
    prob = _prob()
    c = crowd_score_constraint(prob, 1, 1, pct=22.0)
    assert not c.is_longshot_floor
    assert c.lo_pct == 21.5
    assert c.hi_pct == 22.5


def test_sparse_crowd_matrix_no_normalization() -> None:
    prob = _prob()
    cells = {"1_1": 30.0, "2_1": 28.0, "1_2": 14.0}
    top3 = frozenset({"1_1", "2_1", "1_2"})
    crowd = build_crowd_matrix_from_cells(
        cells, prob=prob, top3_keys=top3, fill_from_p=False
    )
    assert crowd.grid_size == 6
    assert crowd.source == "sparse_cells"
    total = sum(sum(row) for row in crowd.matrix)
    assert abs(total - 0.72) < 1e-6
    assert crowd.known is not None
    assert crowd.known[1][1]
    assert not crowd.known[0][0]


def test_p_fill_distributes_remaining_by_p_ratios() -> None:
    prob = _prob()
    cells = {"1_1": 30.0, "2_1": 28.0, "1_2": 14.0}
    top3 = frozenset({"1_1", "2_1", "1_2"})
    crowd = build_crowd_matrix_from_cells(cells, prob=prob, top3_keys=top3)
    assert crowd.grid_size == 6
    assert crowd.estimated is not None
    assert any(any(row) for row in crowd.estimated)
    total_pct = sum(sum(row) for row in crowd.matrix) * 100.0
    assert abs(total_pct - 100.0) < 0.5
    assert crowd.known is not None and not crowd.known[0][0]
    assert crowd.estimated[0][0]


def test_merge_api_top3_preserves_manual() -> None:
    cells = {"1_1": 25.0}
    merged = merge_api_top3_into_cells(cells, {"1:1": 30, "2:1": 28})
    assert merged["1_1"] == 25.0
    assert merged["2_1"] == 28.0


def test_manual_entry_recalculates_fill() -> None:
    prob = _prob()
    cells = {"1_1": 30.0, "2_1": 28.0, "1_2": 14.0, "0_0": 5.0}
    crowd = build_crowd_matrix_from_cells(cells, prob=prob)
    assert crowd.known is not None and crowd.known[0][0]
    total_pct = sum(sum(row) for row in crowd.matrix) * 100.0
    assert abs(total_pct - 100.0) < 0.5
    # Remaining 23% (not 28%) distributed among non-entered cells
    assert crowd.estimated is not None and crowd.estimated[0][1]


def test_empty_cells_matrix() -> None:
    crowd = build_crowd_matrix_from_cells({})
    assert crowd.known is not None
    assert not any(any(row) for row in crowd.known)


def test_empty_cells_fill_from_p_when_prob_available() -> None:
    prob = _prob()
    crowd = build_crowd_matrix_from_cells({}, prob=prob)
    assert crowd.estimated is not None
    assert any(any(row) for row in crowd.estimated)
    total_pct = sum(sum(row) for row in crowd.matrix) * 100.0
    assert abs(total_pct - 100.0) < 0.5
