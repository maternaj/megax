"""Tests for background job progress store."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from megax.gui.jobs import JobStore, recover_job_payload
from megax.gui.state import OptimizerSnapshot, RoundGuiState
from megax.storage import RoundRecord, save_round_record


def test_job_store_progress_and_eta() -> None:
    store = JobStore()
    store.start("round-a", kind="optimize", total=100.0, message="start", redirect_extra="optimized=1")
    store.update("round-a", done=25.0, message="quarter")
    job = store.get("round-a")
    assert job is not None
    assert job.percent() == 25.0
    assert job.message == "quarter"
    time.sleep(0.05)
    payload = job.to_dict()
    assert payload["running"] is True
    assert payload["eta_seconds"] is not None
    store.complete("round-a", message="done")
    done = store.get("round-a")
    assert done is not None
    assert done.status == "done"
    assert done.percent() == 100.0


def test_job_store_rejects_duplicate_running() -> None:
    store = JobStore()
    store.start("round-b", kind="simulate", total=10.0, message="go", redirect_extra="sim=1")
    try:
        store.start("round-b", kind="simulate", total=10.0, message="again", redirect_extra="sim=1")
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_finished_job_survives_long_runtime(monkeypatch) -> None:
    clock = {"now": 0.0}

    def monotonic() -> float:
        return clock["now"]

    monkeypatch.setattr("megax.gui.jobs.time.monotonic", monotonic)

    store = JobStore()
    store.start("round-long", kind="optimize", total=10.0, message="start", redirect_extra="optimized=1")
    clock["now"] = 900.0
    store.complete("round-long", message="done")
    clock["now"] = 950.0
    job = store.get("round-long")
    assert job is not None
    assert job.status == "done"
    clock["now"] = 1300.0
    assert store.get("round-long") is None


def test_recover_job_payload_from_saved_optimization(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("megax.storage.rounds_data_dir", lambda: tmp_path)
    now = datetime.now(timezone.utc)
    state = RoundGuiState(
        pending_mc_job="optimize",
        pending_mc_started_at=(now.replace(microsecond=0).isoformat()),
        last_optimization=OptimizerSnapshot(
            metric="top10",
            objective=0.02,
            tips_a={"1": "1:0"},
            tips_b={"1": "0:0"},
            joker_a=1,
            joker_b=1,
            p_win_a=0.01,
            p_top_10_a=0.02,
            p_top_100_a=0.05,
            mean_pts_a=10.0,
            p_win_b=0.01,
            p_top_10_b=0.02,
            p_top_100_b=0.05,
            mean_pts_b=10.0,
            universes=100,
            crowd_players=100,
            field_size=1000,
            optimized_at=now.isoformat(),
            note="recovered",
        ),
    )
    save_round_record(
        RoundRecord(
            round_key="round_99",
            state=state,
            matches=(),
            saved_at=now,
        )
    )

    payload = recover_job_payload("round_99", "optimize")
    assert payload is not None
    assert payload["status"] == "done"
    assert payload["recovered"] is True
    assert payload["redirect_extra"] == "optimized=1"
