"""Tests for Tipsport offer parsing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from megax.tipsport.offer import (
    filter_matches_by_kickoff_window,
    group_by_kickoff_slot,
    parse_match,
)

FIXTURE = Path(__file__).parent / "fixtures" / "comp120_match_allevents.json"


def _load_match() -> dict:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["matches"][0]


def test_parse_match_extracts_1x2_and_total_25() -> None:
    match = parse_match(_load_match())
    assert match is not None
    assert match.match_id == 7154537
    assert match.home == "Ostrava"
    assert match.away == "Slavia Praha"
    assert match.odds.home == 5.43
    assert match.odds.draw == 4.19
    assert match.odds.away == 1.61
    assert match.odds.over_2_5 == 1.75
    assert match.odds.under_2_5 == 2.1
    assert len(match.odds.home_team_lines) >= 2
    assert len(match.odds.away_team_lines) >= 2
    assert len(match.odds.match_total_lines) >= 3


def test_group_by_kickoff_slot() -> None:
    match = parse_match(_load_match())
    assert match is not None
    duplicate = match
    slots = group_by_kickoff_slot([match, duplicate])
    assert len(slots) == 1
    assert len(slots[0].matches) == 2


def test_filter_matches_by_kickoff_window() -> None:
    match = parse_match(_load_match())
    assert match is not None
    start = datetime(2026, 4, 5, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 6, 0, 0, tzinfo=timezone.utc)
    filtered = filter_matches_by_kickoff_window([match], date_from=start, date_to=end)
    assert len(filtered) == 1

    outside = filter_matches_by_kickoff_window(
        [match],
        date_from=datetime(2026, 4, 1, tzinfo=timezone.utc),
        date_to=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )
    assert outside == []
