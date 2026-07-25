"""Tests for calibration snapshot persistence."""

from __future__ import annotations

from megax.gui.state import CalibrationSnapshot, RoundGuiState


def test_calibration_snapshot_roundtrip() -> None:
    snap = CalibrationSnapshot(
        gpp_ev_ratio=0.85,
        alpha_multiplier=1.0,
        leverage_count=0,
        alpha_used=0.96,
        p_win_best=0.028,
        p_win_a=0.028,
        p_win_b=0.011,
        p_win_pure_ev_joker=0.028,
        use_chalk_mode=True,
        calibrated_at="2026-07-25T08:00:00+00:00",
        universes=1500,
        grid_size=12,
    )
    state = RoundGuiState(calibration=snap)
    restored = RoundGuiState.from_dict(state.to_dict())
    assert restored.calibration is not None
    assert restored.calibration.leverage_count == 0
    assert restored.calibration.use_chalk_mode is True
