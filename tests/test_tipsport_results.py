"""Tests for Tipsport results API parsing."""

from __future__ import annotations

import json
from pathlib import Path

from megax.tipsport.results import (
    MatchStatus,
    match_has_results,
    parse_ft_score,
    parse_match_result,
    parse_result_cells,
)

FIXTURE = Path(__file__).parent / "fixtures" / "match_results_sample.json"


def test_match_has_results() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert match_has_results(data)
    assert not match_has_results({"match": {}})


def test_parse_ft_score() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert parse_ft_score(data) == (1, 2)


def test_parse_result_cells() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cells = parse_result_cells(data)
    assert set(cells) == {2567010869, 2567010868, 9001, 9002, 9003, 9004}
    assert cells[2567010869].winning is True
    assert cells[2567010869].odd == 2.14


def test_parse_match_result() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = parse_match_result(data)
    assert result is not None
    assert result.match_id == 7765938
    assert result.status == MatchStatus.FINISHED
    assert result.home_goals == 1
    assert result.away_goals == 2
