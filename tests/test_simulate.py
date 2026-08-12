"""Tests for Monte Carlo simulation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from megax.gui.state import RoundGuiState
from tests.crowd_fixtures import apply_sample_crowd
from megax.lineup import build_round_lineup
from megax.simulate import (
    SimulationConfig,
    build_default_agents,
    build_lineup_contexts,
    build_match_sim_contexts,
    format_simulation_report,
    run_simulation,
    sample_score,
    score_round,
    simulate_round_record,
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
    apply_sample_crowd(state, match.match_id)
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


def test_run_simulation_reports_progress(tmp_path, monkeypatch) -> None:
    from test_probability import _plzen_match

    monkeypatch.setattr("megax.storage.rounds_data_dir", lambda: tmp_path)
    match = _plzen_match()
    state = RoundGuiState(field_size=5_000)
    state.ensure_match(match.match_id)
    apply_sample_crowd(state, match.match_id)
    contexts = build_match_sim_contexts((match,), state)
    agents = build_default_agents(contexts, lineup=None, state=None)
    calls: list[tuple[int, int]] = []

    run_simulation(
        contexts,
        agents,
        sim_config=SimulationConfig(universes=200, field_size=5_000, crowd_players=50, seed=42),
        progress=lambda done, total: calls.append((done, total)),
    )

    assert calls
    assert calls[-1] == (200, 200)
    assert all(done <= total for done, total in calls)


def test_build_default_agents_includes_pure_ev_joker_with_lineup() -> None:
    from test_probability import _plzen_match

    match = _plzen_match()
    state = RoundGuiState(field_size=5_000)
    state.ensure_match(match.match_id)
    apply_sample_crowd(state, match.match_id)
    contexts = build_match_sim_contexts((match,), state)
    lineup = build_round_lineup(build_lineup_contexts((match,), state))
    agents = build_default_agents(contexts, lineup=lineup, state=None)
    by_name = {agent.name: agent for agent in agents}

    assert "pure_ev_joker" in by_name
    assert by_name["pure_ev_joker"].tips == by_name["pure_ev"].tips
    assert by_name["pure_ev_joker"].joker_match_id == lineup.account_a.joker_match_id
    assert by_name["pure_ev"].joker_match_id is None


def test_points_lut_matches_scoring_rules() -> None:
    from megax.simulate_engine import build_points_lut, flat_index
    from megax.scoring import points

    lut = build_points_lut()
    for tip_home in range(3):
        for tip_away in range(3):
            for actual_home in range(3):
                for actual_away in range(3):
                    tip_flat = flat_index(tip_home, tip_away)
                    actual_flat = flat_index(actual_home, actual_away)
                    assert lut[tip_flat, actual_flat] == points(
                        tip_home, tip_away, actual_home, actual_away
                    )


def test_vectorized_scoring_matches_score_round() -> None:
    from megax.simulate_engine import build_points_lut, flat_index

    lut = build_points_lut()
    tips = {1: "2:1", 2: "1:0", 3: "0:2"}
    outcomes = {1: (2, 1), 2: (1, 1), 3: (0, 2)}
    match_ids = (1, 2, 3)
    expected = score_round(
        tips=tips,
        outcomes=outcomes,
        match_ids=match_ids,
        joker_match_id=1,
    )

    tip_flats = []
    joker_mult = []
    actual_flats = []
    for match_id in match_ids:
        parsed = __import__("megax.ev", fromlist=["parse_tip"]).parse_tip(tips[match_id])
        assert parsed is not None
        tip_flats.append(flat_index(parsed[0], parsed[1]))
        joker_mult.append(2 if match_id == 1 else 1)
        actual = outcomes[match_id]
        actual_flats.append(flat_index(actual[0], actual[1]))

    total = sum(
        lut[tip_flats[idx], actual_flats[idx]] * joker_mult[idx]
        for idx in range(len(match_ids))
    )
    assert total == expected


def test_simulate_skips_matches_without_p_matrix() -> None:
    from test_probability import _plzen_match

    from megax.tipsport.offer import MatchOdds, MegaxMatch

    playable = _plzen_match()
    postponed = MegaxMatch(
        match_id=999999,
        name="Odložený - TBD",
        home="Odložený",
        away="TBD",
        kickoff_at=playable.kickoff_at,
        odds=MatchOdds(home=0.0, draw=0.0, away=0.0),
        match_type="MATCH_POSTPONED",
        ended=False,
        competition_id=playable.competition_id,
    )
    state = RoundGuiState(field_size=5_000)
    for match in (playable, postponed):
        state.ensure_match(match.match_id)
        apply_sample_crowd(state, match.match_id)
    record = RoundRecord(
        round_key="sim-skip-postponed",
        state=state,
        matches=(playable, postponed),
        saved_at=datetime.now(timezone.utc),
    )
    result = simulate_round_record(
        record,
        sim_config=SimulationConfig(
            universes=20,
            field_size=5_000,
            crowd_players=30,
            seed=7,
        ),
        include_saved_agents=False,
    )
    assert result.skipped_match_ids == (postponed.match_id,)
    assert len(result.matches) == 1
    assert result.matches[0].match_id == playable.match_id
    assert result.agents


def test_load_and_simulate_missing_round() -> None:
    from megax.simulate import load_and_simulate

    with pytest.raises(FileNotFoundError):
        load_and_simulate("missing-round-key")


def test_simulate_optimizer_uses_calibrated_knobs() -> None:
    from megax.calibrate import build_lineup_for_knobs, knobs_from_snapshot
    from megax.storage import load_round_record

    record = load_round_record("2026-07-24_2026-07-27")
    if record is None or record.state.calibration is None:
        pytest.skip("Round snapshot with calibration required")
    if not record.state.crowd_cells:
        pytest.skip("Round snapshot needs crowd_cells for sparse C model")

    calibrated = build_lineup_for_knobs(
        record.matches,
        record.state,
        knobs_from_snapshot(record.state.calibration),
    )
    assert calibrated is not None

    result = simulate_round_record(
        record,
        sim_config=SimulationConfig(
            universes=50,
            field_size=record.state.field_size,
            crowd_players=40,
            seed=1,
        ),
        include_saved_agents=False,
    )
    assert result.optimizer_note.startswith("calibrated")
    optimizer_a = next(agent for agent in result.agents if agent.name == "optimizer_a")
    optimizer_b = next(agent for agent in result.agents if agent.name == "optimizer_b")
    assert optimizer_a.tips == calibrated.account_a.tips_by_match()
    assert optimizer_b.tips == calibrated.account_b.tips_by_match()


def test_format_simulation_report_includes_tips(tmp_path, monkeypatch) -> None:
    from test_probability import _plzen_match

    monkeypatch.setattr("megax.storage.rounds_data_dir", lambda: tmp_path)
    match = _plzen_match()
    state = RoundGuiState(field_size=5_000)
    state.ensure_match(match.match_id)
    apply_sample_crowd(state, match.match_id)
    record = RoundRecord(
        round_key="report-sim",
        state=state,
        matches=(match,),
        saved_at=datetime.now(timezone.utc),
    )
    result = simulate_round_record(
        record,
        sim_config=SimulationConfig(universes=50, field_size=5_000, crowd_players=20, seed=1),
        include_saved_agents=False,
    )
    report = format_simulation_report(result)
    assert "Tips by agent" in report
    assert "pure_ev" in report
    assert match.name.replace(" - ", "–") in report or match.name in report
