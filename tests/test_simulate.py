"""Tests for Monte Carlo simulation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from megax.gui.state import RoundGuiState
from megax.simulate import (
    SimulationConfig,
    build_default_agents,
    build_match_sim_contexts,
    run_simulation,
    sample_score,
    score_round,
)
from megax.storage import RoundRecord, save_round_record


def test_sample_score_returns_valid_cell() -> None:
    matrix = (
        (0.5, 0.3),
        (0.1, 0.1),
    )
    rng = __import__("random").Random(0)
    home, away = sample_score(rng, matrix)
    assert 0 <= home < 2
    assert 0 <= away < 2


def test_score_round_with_joker() -> None:
    scored = score_round(
        tips={1: "2:1"},
        outcomes={1: (2, 1)},
        match_ids=(1,),
        joker_match_id=1,
    )
    assert scored == 20


def test_run_simulation_produces_stats(tmp_path, monkeypatch) -> None:
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
        round_key="test-round",
        state=state,
        matches=(match,),
        saved_at=datetime.now(timezone.utc),
    )
    save_round_record(record)

    contexts = build_match_sim_contexts(record.matches, record.state)
    agents = build_default_agents(contexts, lineup=None, state=None)
    result = run_simulation(
        contexts,
        agents,
        sim_config=SimulationConfig(universes=200, field_size=5_000, crowd_players=50, seed=42),
    )
    assert result.universes == 200
    assert len(result.agents) == 2
    assert all(agent.mean_points >= 0 for agent in result.agents)
    assert all(0.0 <= agent.p_win <= 1.0 for agent in result.agents)


def test_load_and_simulate_missing_round() -> None:
    from megax.simulate import load_and_simulate

    with pytest.raises(FileNotFoundError):
        load_and_simulate("missing-round-key")
