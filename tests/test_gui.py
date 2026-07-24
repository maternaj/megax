"""Tests for MegaX GUI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from megax.gui.app import create_app
from megax.storage import load_round_record
from megax.gui.weekend import round_key
from megax.ingest import RoundSnapshot
from megax.tipsport.offer import KickoffSlot, parse_match

FIXTURE = Path(__file__).parent / "fixtures" / "comp120_match_allevents.json"


@pytest.fixture
def mock_round(monkeypatch):
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    match = parse_match(raw["matches"][0])
    assert match is not None
    date_from = datetime(2026, 4, 5, 0, 0, tzinfo=timezone.utc)
    date_to = datetime(2026, 4, 6, 0, 0, tzinfo=timezone.utc)
    snapshot = RoundSnapshot(
        competition_id=120,
        date_from=date_from,
        date_to=date_to,
        fetched_at=datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc),
        matches=(match,),
        slots=(KickoffSlot(kickoff_at=match.kickoff_at, matches=(match,)),),
    )

    client = MagicMock()
    client.fetch.return_value = raw

    with patch("megax.gui.service.fetch_round_snapshot", return_value=snapshot), patch(
        "megax.gui.service.poll_once",
        return_value=type(
            "Poll",
            (),
            {"polled_at": datetime(2026, 4, 5, 12, 5, tzinfo=timezone.utc), "results": {match.match_id: None}},
        )(),
    ), patch("megax.gui.app.fetch_round_snapshot", return_value=snapshot), patch(
        "megax.gui.app._tipsport_client", return_value=client
    ):
        yield snapshot, match


def test_home_renders(mock_round) -> None:
    snapshot, match = mock_round
    app = create_app()
    response = TestClient(app).get("/?from_day=2026-04-05&to_day=2026-04-06")
    assert response.status_code == 200
    body = response.text
    assert "MegaX" in body
    assert match.name in body
    assert "Peníze" in body
    assert "EV tip" in body
    assert "GPP tip" in body
    assert "Auto-refresh" in body
    assert 'id="auto_refresh"' in body
    assert "Vyplnit tipy A/B" in body
    assert "P(x,y)" in body


def test_save_round_state(mock_round, tmp_path, monkeypatch) -> None:
    snapshot, match = mock_round
    monkeypatch.setattr("megax.storage.rounds_data_dir", lambda: tmp_path)
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/save",
        data={
            "from_day": "2026-04-05",
            "to_day": "2026-04-06",
            "field_size": "42000",
            "rank_a": "100",
            "points_a": "12",
            "rank_b": "250",
            "points_b": "8",
            f"tip_a_{match.match_id}": "2:1",
            f"tip_b_{match.match_id}": "1:1",
            f"money_{match.match_id}_tipsport_home": "55",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    key = round_key(snapshot.date_from, snapshot.date_to)
    record = load_round_record(key)
    assert record is not None
    assert record.state.field_size == 42000
    assert record.state.accounts["A"].tips[str(match.match_id)] == "2:1"
    assert record.state.money[str(match.match_id)]["tipsport"]["home"] == 55.0
    assert len(record.matches) == 1


def test_default_weekend_window_friday_to_monday() -> None:
    from megax.gui.weekend import PRAGUE, default_round_window

    now = datetime(2026, 7, 24, 10, 0, tzinfo=timezone.utc)  # Friday noon CEST
    date_from, date_to = default_round_window(now)
    assert date_from.astimezone(PRAGUE).date().isoformat() == "2026-07-24"
    assert date_to.astimezone(PRAGUE).date().isoformat() == "2026-07-27"
