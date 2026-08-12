"""Tests for late-swap state machine."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from megax.gui.state import RoundGuiState
from tests.crowd_fixtures import apply_sample_crowd
from megax.ingest import RoundSnapshot
from megax.swap import (
    SwapConfig,
    SwapMode,
    apply_swap_to_state,
    classify_swap_mode,
    compute_swap_recommendation,
    estimate_leader_points,
)
from megax.simulate import build_lineup_contexts
from megax.tipsport.offer import group_by_kickoff_slot


def test_classify_swap_mode() -> None:
    cfg = SwapConfig(delta_small=3, delta_large=8)
    assert classify_swap_mode(2.0, cfg) == SwapMode.PROTECT
    assert classify_swap_mode(5.0, cfg) == SwapMode.NEUTRAL
    assert classify_swap_mode(10.0, cfg) == SwapMode.CHASE


def test_compute_swap_recommendation_for_future_slots() -> None:
    from test_probability import _plzen_match

    base = datetime(2026, 7, 25, 15, 0, tzinfo=timezone.utc)
    plzen = _plzen_match()
    teplice = replace(
        plzen,
        match_id=8212281,
        name="Baník - Teplice",
        kickoff_at=base + timedelta(hours=26),
    )
    matches = (plzen, teplice)
    state = RoundGuiState(field_size=10_000)
    for match in matches:
        state.ensure_match(match.match_id)
        apply_sample_crowd(state, match.match_id)
        state.accounts["A"].tips[str(match.match_id)] = "0:0"
        state.accounts["B"].tips[str(match.match_id)] = "0:0"

    snapshot = RoundSnapshot(
        competition_id=120,
        date_from=base,
        date_to=base + timedelta(days=3),
        fetched_at=base,
        matches=matches,
        slots=tuple(group_by_kickoff_slot(list(matches))),
    )
    contexts = build_lineup_contexts(matches, state)
    now = base + timedelta(hours=1)
    swap = compute_swap_recommendation(
        snapshot=snapshot,
        state=state,
        contexts=contexts,
        results={},
        now=now,
    )
    assert swap is not None
    assert swap.remaining_match_count == 1
    assert swap.next_slot_at == teplice.kickoff_at
    assert swap.mode in (SwapMode.PROTECT, SwapMode.NEUTRAL, SwapMode.CHASE)


def test_apply_swap_updates_only_remaining_tips() -> None:
    from test_probability import _plzen_match

    base = datetime(2026, 7, 25, 15, 0, tzinfo=timezone.utc)
    plzen = _plzen_match()
    teplice = replace(
        plzen,
        match_id=8212281,
        name="Baník - Teplice",
        kickoff_at=base + timedelta(hours=26),
    )
    matches = (plzen, teplice)
    state = RoundGuiState(field_size=10_000)
    for match in matches:
        state.ensure_match(match.match_id)
        apply_sample_crowd(state, match.match_id)
        state.accounts["A"].tips[str(match.match_id)] = "0:0"
        state.accounts["B"].tips[str(match.match_id)] = "0:0"

    snapshot = RoundSnapshot(
        competition_id=120,
        date_from=base,
        date_to=base + timedelta(days=3),
        fetched_at=base,
        matches=matches,
        slots=tuple(group_by_kickoff_slot(list(matches))),
    )
    contexts = build_lineup_contexts(matches, state)
    swap = compute_swap_recommendation(
        snapshot=snapshot,
        state=state,
        contexts=contexts,
        results={},
        now=base + timedelta(hours=1),
    )
    assert swap is not None
    locked_tip = state.accounts["A"].tips[str(plzen.match_id)]
    apply_swap_to_state(state, swap, remaining_match_ids=set(swap.remaining_match_ids))
    assert state.accounts["A"].tips[str(plzen.match_id)] == locked_tip


def test_compute_swap_recommendation_before_first_slot_returns_none() -> None:
    from test_probability import _plzen_match

    base = datetime(2026, 7, 25, 8, 0, tzinfo=timezone.utc)
    plzen = replace(_plzen_match(), kickoff_at=datetime(2026, 7, 25, 15, 0, tzinfo=timezone.utc))
    matches = (plzen,)
    state = RoundGuiState(field_size=10_000)
    state.ensure_match(plzen.match_id)
    apply_sample_crowd(state, plzen.match_id)
    snapshot = RoundSnapshot(
        competition_id=120,
        date_from=base,
        date_to=base + timedelta(days=3),
        fetched_at=base,
        matches=matches,
        slots=tuple(group_by_kickoff_slot(list(matches))),
    )
    contexts = build_lineup_contexts(matches, state)
    swap = compute_swap_recommendation(
        snapshot=snapshot,
        state=state,
        contexts=contexts,
        results={},
        now=base,
    )
    assert swap is None


def test_estimate_leader_points() -> None:
    from test_probability import _plzen_match

    match = _plzen_match()
    state = RoundGuiState(field_size=10_000)
    state.ensure_match(match.match_id)
    apply_sample_crowd(state, match.match_id)
    contexts = build_lineup_contexts((match,), state)
    leader = estimate_leader_points(contexts, SwapConfig(leader_chalk=0.85))
    assert leader > 0
