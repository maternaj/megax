"""Tests for simulate-driven calibration."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from megax.calibrate import (
    CalibrationKnobs,
    default_knob_grid,
    format_calibration_report,
    load_and_calibrate,
    resolve_calibration_alpha,
)
from megax.gui.state import RoundGuiState
from megax.simulate import SimulationConfig
from megax.storage import RoundRecord, save_round_record


def test_resolve_calibration_alpha_scales_with_multiplier() -> None:
    base = resolve_calibration_alpha(10_000, alpha_multiplier=1.0)
    higher = resolve_calibration_alpha(10_000, alpha_multiplier=1.2)
    assert higher > base


def test_default_knob_grid_quick_is_smaller() -> None:
    full = default_knob_grid(8, quick=False)
    quick = default_knob_grid(8, quick=True)
    assert len(quick) < len(full)
    assert all(isinstance(knob, CalibrationKnobs) for knob in quick)


def test_load_and_calibrate_runs_grid(tmp_path, monkeypatch) -> None:
    from test_probability import _plzen_match

    monkeypatch.setattr("megax.storage.rounds_data_dir", lambda: tmp_path)
    match = _plzen_match()
    state = RoundGuiState(field_size=5_000)
    state.ensure_match(match.match_id)
    state.money[str(match.match_id)] = {
        "tipsport": {"home": 70, "draw": 15, "away": 15, "over": 50, "under": 50},
        "fortuna": {"home": 75, "draw": 10, "away": 15, "over": 55, "under": 45},
        "sazkabet": {"home": 72, "draw": 12, "away": 16, "over": 52, "under": 48},
    }
    record = RoundRecord(
        round_key="cal-test",
        state=state,
        matches=(match,),
        saved_at=datetime.now(timezone.utc),
    )
    save_round_record(record)

    grid = (
        CalibrationKnobs(gpp_ev_ratio=0.85, alpha_multiplier=0.85, leverage_count=0),
        CalibrationKnobs(gpp_ev_ratio=0.85, alpha_multiplier=1.15, leverage_count=0),
    )
    result = load_and_calibrate(
        "cal-test",
        sim_config=SimulationConfig(universes=100, field_size=5_000, crowd_players=30, seed=42),
        grid=grid,
    )
    assert len(result.rows) >= 1
    assert result.best.p_win_best >= 0.0
    report = format_calibration_report(result)
    assert "Recommendation" in report
    assert "Top 10" in report


def test_load_and_calibrate_missing_round() -> None:
    with pytest.raises(FileNotFoundError):
        load_and_calibrate("missing-round")
