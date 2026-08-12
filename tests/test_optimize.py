"""Tests for Monte Carlo lineup optimizer."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from megax.gui.state import OptimizerSnapshot, RoundGuiState
from megax.optimize import (
    DualLineup,
    apply_optimizer_snapshot,
    build_optimize_cache,
    gui_optimize_config,
    gui_optimize_config_pair,
    objective_value,
    optimize_dual_lineup,
    optimize_round_record,
    optimizer_snapshot_from_result,
    score_dual_lineup,
    tip_candidates,
)
from megax.simulate import SimulationConfig, build_match_sim_contexts
from megax.storage import RoundRecord
from tests.crowd_fixtures import apply_sample_crowd


def test_tip_candidates_includes_ev_best() -> None:
    from test_probability import _plzen_match

    match = _plzen_match()
    state = RoundGuiState(field_size=5_000)
    state.ensure_match(match.match_id)
    apply_sample_crowd(state, match.match_id)
    contexts = build_match_sim_contexts((match,), state)
    assert contexts
    candidates = tip_candidates(contexts[0], top_n=3)
    assert candidates
    assert contexts[0].analysis.ev.best.score in candidates


def test_objective_value_averages_accounts() -> None:
    from megax.optimize import AccountOptimizeStats

    stats = AccountOptimizeStats(
        mean_points=10.0,
        p_win=0.02,
        p_top_10=0.20,
        p_top_100=0.05,
        p_top_1000=0.01,
    )
    assert objective_value(stats, stats, "top10") == pytest.approx(0.20)
    assert objective_value(stats, stats, "top1") == pytest.approx(0.05)
    assert objective_value(stats, stats, "win") == pytest.approx(0.02)


def test_estimate_optimize_evaluations_positive() -> None:
    from test_probability import _plzen_match

    match = _plzen_match()
    state = RoundGuiState(field_size=5_000)
    state.ensure_match(match.match_id)
    apply_sample_crowd(state, match.match_id)
    contexts = build_match_sim_contexts((match,), state)
    from megax.optimize import estimate_optimize_evaluations, estimate_optimize_units

    assert estimate_optimize_evaluations(contexts) >= 3
    _budget, units = estimate_optimize_units(contexts, universes=200)
    assert units >= 200


def test_optimize_reports_progress() -> None:
    from test_probability import _plzen_match

    match = _plzen_match()
    state = RoundGuiState(field_size=5_000)
    state.ensure_match(match.match_id)
    apply_sample_crowd(state, match.match_id)
    contexts = build_match_sim_contexts((match,), state)
    search_cfg, final_cfg = gui_optimize_config_pair(
        state.field_size,
        universes=60,
        crowd_players=20,
        search_universes=40,
        seed=1,
    )
    calls: list[tuple[str, float, float]] = []

    def on_progress(phase: str, done: float, total: float, detail: str) -> None:
        calls.append((phase, done, total))

    optimize_dual_lineup(
        contexts,
        search_config=search_cfg,
        final_config=final_cfg,
        max_passes=1,
        progress=on_progress,
    )
    assert calls
    assert any(phase == "sample" for phase, _, _ in calls)
    assert any(phase == "search" for phase, _, _ in calls)


def test_shared_universe_rescore_matches_full_simulation() -> None:
    from test_probability import _plzen_match

    from megax.simulate import AgentSpec, run_simulation

    match = _plzen_match()
    state = RoundGuiState(field_size=5_000)
    state.ensure_match(match.match_id)
    apply_sample_crowd(state, match.match_id)
    contexts = build_match_sim_contexts((match,), state)
    cfg = SimulationConfig(universes=300, field_size=5_000, crowd_players=40, seed=99)
    cache = build_optimize_cache(contexts, sim_config=cfg)
    lineup = DualLineup(
        tips_a={match.match_id: contexts[0].analysis.ev.best.score},
        tips_b={match.match_id: contexts[0].analysis.gpp_best.score},
        joker_a=match.match_id,
        joker_b=match.match_id,
    )
    fast_a, fast_b = score_dual_lineup(cache, lineup)
    agents = (
        AgentSpec(name="mc_opt_a", tips=lineup.tips_a, joker_match_id=lineup.joker_a),
        AgentSpec(name="mc_opt_b", tips=lineup.tips_b, joker_match_id=lineup.joker_b),
    )
    full = run_simulation(contexts, agents, sim_config=cfg)
    by_name = {agent.name: agent for agent in full.agents}
    assert fast_a.p_top_10 == pytest.approx(by_name["mc_opt_a"].p_top_10, abs=1e-12)
    assert fast_b.p_win == pytest.approx(by_name["mc_opt_b"].p_win, abs=1e-12)


def test_optimize_dual_lineup_runs(tmp_path, monkeypatch) -> None:
    from test_probability import _plzen_match

    monkeypatch.setattr("megax.storage.rounds_data_dir", lambda: tmp_path)
    match = _plzen_match()
    state = RoundGuiState(field_size=5_000)
    state.ensure_match(match.match_id)
    apply_sample_crowd(state, match.match_id)
    contexts = build_match_sim_contexts((match,), state)
    search_cfg, final_cfg = gui_optimize_config_pair(
        state.field_size,
        universes=100,
        crowd_players=30,
        search_universes=50,
        seed=42,
    )
    result = optimize_dual_lineup(
        contexts,
        metric="top10",
        search_config=search_cfg,
        final_config=final_cfg,
        max_passes=1,
    )
    assert result.lineup.tips_a
    assert result.lineup.tips_b
    assert result.search_evaluations >= 1
    assert 0.0 <= result.objective <= 1.0


def test_optimize_round_record_skips_postponed() -> None:
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
        round_key="opt-skip-postponed",
        state=state,
        matches=(playable, postponed),
        saved_at=datetime.now(timezone.utc),
    )
    search_cfg, final_cfg = gui_optimize_config_pair(
        state.field_size,
        universes=40,
        crowd_players=20,
        search_universes=30,
        seed=3,
    )
    result = optimize_round_record(
        record,
        search_config=search_cfg,
        final_config=final_cfg,
    )
    assert postponed.match_id in result.skipped_match_ids
    assert len(result.lineup.tips_a) == 1


def test_optimizer_snapshot_roundtrip() -> None:
    from test_probability import _plzen_match

    match = _plzen_match()
    state = RoundGuiState(field_size=5_000)
    state.ensure_match(match.match_id)
    apply_sample_crowd(state, match.match_id)
    record = RoundRecord(
        round_key="opt-snapshot",
        state=state,
        matches=(match,),
        saved_at=datetime.now(timezone.utc),
    )
    search_cfg, final_cfg = gui_optimize_config_pair(
        state.field_size,
        universes=40,
        crowd_players=20,
        search_universes=30,
        seed=5,
    )
    result = optimize_round_record(
        record,
        search_config=search_cfg,
        final_config=final_cfg,
    )
    snap = optimizer_snapshot_from_result(result)
    restored = OptimizerSnapshot.from_dict(snap.to_dict())
    assert restored.metric == snap.metric
    assert restored.tips_a == snap.tips_a
    assert restored.joker_a == snap.joker_a
    assert restored.objective == snap.objective


def test_apply_optimizer_snapshot_writes_accounts() -> None:
    state = RoundGuiState(field_size=5_000)
    snap = OptimizerSnapshot(
        metric="top10",
        objective=0.15,
        tips_a={"101": "2:1", "102": "1:0"},
        tips_b={"101": "1:1", "102": "0:2"},
        joker_a=101,
        joker_b=102,
        p_win_a=0.01,
        p_top_10_a=0.15,
        p_top_100_a=0.04,
        mean_pts_a=8.0,
        p_win_b=0.01,
        p_top_10_b=0.14,
        p_top_100_b=0.03,
        mean_pts_b=7.5,
        universes=100,
        crowd_players=50,
        field_size=5_000,
    )
    apply_optimizer_snapshot(state, snap)
    assert state.accounts["A"].tips["101"] == "2:1"
    assert state.accounts["B"].tips["102"] == "0:2"
    assert state.accounts["A"].joker_match_id == 101
    assert state.accounts["B"].joker_match_id == 102
