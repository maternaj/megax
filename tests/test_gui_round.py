"""Tests for round-id centric GUI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from megax.gui.app import create_app
from megax.gui.megatip_bridge import MegatipFetchResult
from megax.ingest import RoundSnapshot
from megax.megatip.models import RoundTipsSnapshot
from megax.storage import load_round_record, round_storage_key
from megax.tipsport.offer import KickoffSlot, parse_match

FIXTURE = Path(__file__).parent / "fixtures" / "comp120_match_allevents.json"
MEGATIP_FIXTURE = Path(__file__).parent / "fixtures" / "megatip" / "clients_tips_round383.json"


@pytest.fixture
def mock_round_id_mode(monkeypatch):
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
    megatip_raw = json.loads(MEGATIP_FIXTURE.read_text(encoding="utf-8"))
    from megax.megatip.parse import parse_round_tips

    round_tips = parse_round_tips(megatip_raw, contest_id=161, round_id=383)
    fetch_result = MegatipFetchResult(
        round_id=383,
        round_number=3,
        round_tips=round_tips,
        snapshot=snapshot,
        missing_match_ids=(),
    )

    client = MagicMock()
    with patch("megax.gui.app.fetch_megatip_round", return_value=fetch_result), patch(
        "megax.gui.service.poll_once",
        return_value=type(
            "Poll",
            (),
            {"polled_at": datetime(2026, 4, 5, 12, 5, tzinfo=timezone.utc), "results": {match.match_id: None}},
        )(),
    ), patch("megax.gui.app._tipsport_client", return_value=client):
        yield snapshot, match, fetch_result


def test_home_renders_round_id_mode(mock_round_id_mode, tmp_path, monkeypatch) -> None:
    snapshot, match, _fetch = mock_round_id_mode
    monkeypatch.setattr("megax.storage.rounds_data_dir", lambda: tmp_path)
    app = create_app()
    response = TestClient(app).get("/?round_id=383")
    assert response.status_code == 200
    body = response.text
    assert "roundId" in body
    assert "383" in body
    assert "C(x,y)" in body
    assert "Megatip (veřejné API)" in body
    assert match.name in body


def test_fetch_megatip_endpoint(mock_round_id_mode, tmp_path, monkeypatch) -> None:
    snapshot, match, fetch_result = mock_round_id_mode
    monkeypatch.setattr("megax.storage.rounds_data_dir", lambda: tmp_path)
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/fetch-megatip",
        data={"round_id": "383", "field_size": "50000"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "megatip=1" in response.headers["location"]
    record = load_round_record(round_storage_key(383))
    assert record is not None
    assert record.state.round_id == 383
    assert record.state.megatip is not None
