"""Tests for round ingest orchestration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from megax.ingest import fetch_round_snapshot
from megax.tipsport.offer import parse_match

FIXTURE = Path(__file__).parent / "fixtures" / "comp120_match_allevents.json"


def test_fetch_round_snapshot_uses_client_and_filters_window() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    match = parse_match(raw["matches"][0])
    assert match is not None

    client = MagicMock()
    client.fetch.return_value = raw

    snapshot = fetch_round_snapshot(
        date_from=datetime(2026, 4, 5, 0, 0, tzinfo=timezone.utc),
        date_to=datetime(2026, 4, 6, 0, 0, tzinfo=timezone.utc),
        client=client,
    )
    assert len(snapshot.matches) == 1
    assert len(snapshot.slots) == 1
    assert snapshot.matches[0].match_id == 7154537
