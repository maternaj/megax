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
from tests.crowd_fixtures import crowd_form_fields

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


@pytest.mark.skip(reason="Calibrate endpoint deferred to Phase 3 strategy rework")
def test_calibrate_and_apply_endpoint_persists_round(mock_round, tmp_path, monkeypatch) -> None:
    from megax.calibrate import CalibrationKnobs, CalibrationResult, CalibrationRow

    snapshot, match = mock_round
    monkeypatch.setattr("megax.storage.rounds_data_dir", lambda: tmp_path)
    monkeypatch.setattr("megax.gui.app.load_and_calibrate", lambda *args, **kwargs: CalibrationResult(
        round_key="test",
        match_count=1,
        universes=100,
        crowd_players=30,
        field_size=5000,
        rows=(
            CalibrationRow(
                knobs=CalibrationKnobs(0.85, 1.0, 0),
                alpha_used=0.96,
                p_win_a=0.03,
                p_win_b=0.01,
                p_win_pure_ev_joker=0.03,
                mean_pts_a=28.0,
                mean_pts_b=26.0,
                beats_baseline=True,
            ),
        ),
        best=CalibrationRow(
            knobs=CalibrationKnobs(0.85, 1.0, 0),
            alpha_used=0.96,
            p_win_a=0.03,
            p_win_b=0.01,
            p_win_pure_ev_joker=0.03,
            mean_pts_a=28.0,
            mean_pts_b=26.0,
            beats_baseline=True,
        ),
        baseline_row=None,
        use_chalk_mode=True,
    ))
    app = create_app()
    client = TestClient(app)
    crowd = crowd_form_fields(match.match_id)
    response = client.post(
        "/calibrate-and-apply",
        data={
            "from_day": "2026-04-05",
            "to_day": "2026-04-06",
            "field_size": "50000",
            **crowd,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "calibrated=1" in response.headers["location"]
    key = round_key(snapshot.date_from, snapshot.date_to)
    record = load_round_record(key)
    assert record is not None
    assert record.state.calibration is not None
    assert isinstance(record.state.calibration.use_chalk_mode, bool)


@pytest.mark.skip(reason="Calibrate endpoint deferred to Phase 3 strategy rework")
def test_calibrate_and_apply_endpoint(mock_round, tmp_path, monkeypatch) -> None:
    from megax.calibrate import CalibrationKnobs, CalibrationResult, CalibrationRow

    snapshot, match = mock_round
    monkeypatch.setattr("megax.storage.rounds_data_dir", lambda: tmp_path)
    monkeypatch.setattr("megax.gui.app.load_and_calibrate", lambda *args, **kwargs: CalibrationResult(
        round_key="test",
        match_count=1,
        universes=100,
        crowd_players=30,
        field_size=5000,
        rows=(
            CalibrationRow(
                knobs=CalibrationKnobs(0.85, 1.0, 0),
                alpha_used=0.96,
                p_win_a=0.03,
                p_win_b=0.01,
                p_win_pure_ev_joker=0.03,
                mean_pts_a=28.0,
                mean_pts_b=26.0,
                beats_baseline=True,
            ),
        ),
        best=CalibrationRow(
            knobs=CalibrationKnobs(0.85, 1.0, 0),
            alpha_used=0.96,
            p_win_a=0.03,
            p_win_b=0.01,
            p_win_pure_ev_joker=0.03,
            mean_pts_a=28.0,
            mean_pts_b=26.0,
            beats_baseline=True,
        ),
        baseline_row=None,
        use_chalk_mode=True,
    ))
    app = create_app()
    client = TestClient(app)
    crowd = crowd_form_fields(match.match_id)
    response = client.post(
        "/calibrate-and-apply",
        data={
            "from_day": "2026-04-05",
            "to_day": "2026-04-06",
            "field_size": "50000",
            **crowd,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "calibrated=1" in response.headers["location"]
    key = round_key(snapshot.date_from, snapshot.date_to)
    record = load_round_record(key)
    assert record is not None
    assert record.state.calibration is not None
    assert record.state.calibration.leverage_count == 0
    assert record.state.accounts["A"].tips


def test_home_renders(mock_round) -> None:
    snapshot, match = mock_round
    app = create_app()
    response = TestClient(app).get("/?from_day=2026-04-05&to_day=2026-04-06")
    assert response.status_code == 200
    body = response.text
    assert "MegaX" in body
    assert match.name in body
    assert "Dav %" not in body or "C(x,y)" in body
    assert "EV tip" in body
    assert "EV/C tip" in body
    assert "Monte Carlo" in body
    assert "Auto-refresh" in body
    assert 'id="auto_refresh"' in body
    assert "C(x,y)" in body
    assert "Strategie" not in body or "Monte Carlo" in body
    assert "P(x,y)" in body


def test_simulate_endpoint_persists_result(mock_round, tmp_path, monkeypatch) -> None:
    import time

    from megax.simulate import AgentStats, SimulationResult

    snapshot, match = mock_round
    monkeypatch.setattr("megax.storage.rounds_data_dir", lambda: tmp_path)
    fake_result = SimulationResult(
        universes=500,
        crowd_players=100,
        field_size=50_000,
        agents=(
            AgentStats(
                name="saved_a",
                mean_points=25.0,
                p_win=0.02,
                p_top_10=0.15,
                p_top_100=0.40,
                p_top_1000=0.80,
                tips={match.match_id: "2:1"},
            ),
        ),
        matches=(match,),
    )

    def _fake_simulate(*args, **kwargs):
        return fake_result

    monkeypatch.setattr("megax.gui.app.simulate_round_record", _fake_simulate)
    app = create_app()
    client = TestClient(app)
    crowd = crowd_form_fields(match.match_id)
    response = client.post(
        "/simulate",
        data={
            "from_day": "2026-04-05",
            "to_day": "2026-04-06",
            "field_size": "50000",
            "sim_universes": "500",
            "sim_crowd_players": "100",
            "tip_a_" + str(match.match_id): "2:1",
            **crowd,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "simulating=1" in response.headers["location"]
    key = round_key(snapshot.date_from, snapshot.date_to)
    for _ in range(50):
        record = load_round_record(key)
        if record is not None and record.state.last_simulation is not None:
            break
        time.sleep(0.05)
    record = load_round_record(key)
    assert record is not None
    assert record.state.last_simulation is not None
    assert record.state.last_simulation.universes == 500
    assert record.state.last_simulation.agents[0].p_win == 0.02


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
            f"crowd_{match.match_id}_2_1": "22",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    key = round_key(snapshot.date_from, snapshot.date_to)
    record = load_round_record(key)
    assert record is not None
    assert record.state.field_size == 42000
    assert record.state.accounts["A"].tips[str(match.match_id)] == "2:1"
    assert record.state.crowd_cells[str(match.match_id)]["2_1"] == 22.0
    assert len(record.matches) == 1


def test_default_weekend_window_friday_to_monday() -> None:
    from megax.gui.weekend import PRAGUE, default_round_window

    now = datetime(2026, 7, 24, 10, 0, tzinfo=timezone.utc)  # Friday noon CEST
    date_from, date_to = default_round_window(now)
    assert date_from.astimezone(PRAGUE).date().isoformat() == "2026-07-24"
    assert date_to.astimezone(PRAGUE).date().isoformat() == "2026-07-27"
