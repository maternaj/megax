"""Tests for results polling."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from megax.poll import poll_once
from megax.tipsport.results import MatchStatus, poll_match_results, should_poll_match_results

FIXTURE = Path(__file__).parent / "fixtures" / "match_results_sample.json"


def test_should_poll_match_results() -> None:
    kickoff = datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc)
    delay = timedelta(hours=1, minutes=45)

    assert not should_poll_match_results(
        kickoff,
        now=datetime(2026, 7, 25, 17, 0, tzinfo=timezone.utc),
        min_delay=delay,
    )
    assert not should_poll_match_results(
        kickoff,
        now=datetime(2026, 7, 25, 19, 44, tzinfo=timezone.utc),
        min_delay=delay,
    )
    assert should_poll_match_results(
        kickoff,
        now=datetime(2026, 7, 25, 19, 45, tzinfo=timezone.utc),
        min_delay=delay,
    )


def test_poll_match_results_skips_prematch() -> None:
    client = MagicMock()
    kickoff = datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc)

    results = poll_match_results(
        client,
        [8212280],
        kickoffs={8212280: kickoff},
        min_delay=timedelta(hours=1, minutes=45),
        now=datetime(2026, 7, 25, 19, 44, tzinfo=timezone.utc),
    )
    assert results == {8212280: None}
    client.fetch_match_results.assert_not_called()


def test_poll_once() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    client = MagicMock()
    client.fetch_match_results.return_value = payload

    snapshot = poll_once([7765938], client=client)
    result = snapshot.results[7765938]
    assert result is not None
    assert result.status == MatchStatus.FINISHED
    assert result.home_goals == 1
    assert result.away_goals == 2


def test_poll_once_skips_when_kickoff_too_recent() -> None:
    client = MagicMock()
    kickoff = datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc)

    snapshot = poll_once(
        [8212280],
        kickoffs={8212280: kickoff},
        client=client,
        now=datetime(2026, 7, 25, 19, 44, tzinfo=timezone.utc),
    )
    assert snapshot.results == {8212280: None}
    client.fetch_match_results.assert_not_called()
