"""Shared crowd cell fixtures for tests."""

from __future__ import annotations

from megax.gui.state import RoundGuiState


def sample_crowd_cells() -> dict[str, float]:
    """Typical sparse C cells in GPP-friendly range (0.5–12%)."""
    return {
        "1_1": 8.0,
        "2_1": 7.0,
        "1_0": 6.0,
        "0_1": 5.0,
        "2_0": 4.0,
        "1_2": 3.0,
        "0_0": 2.0,
        "2_2": 2.0,
    }


def apply_sample_crowd(state: RoundGuiState, match_id: int) -> None:
    state.ensure_match(match_id)
    state.crowd_cells[str(match_id)] = sample_crowd_cells()


def crowd_form_fields(match_id: int) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, pct in sample_crowd_cells().items():
        home, away = key.split("_", 1)
        fields[f"crowd_{match_id}_{home}_{away}"] = str(pct)
    return fields
