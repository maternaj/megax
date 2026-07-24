"""Tests for round lineup optimizer."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from megax.crowd import build_crowd_matrix
from megax.gui.state import RoundGuiState
from megax.lineup import (
    MatchLineupContext,
    apply_lineup_to_state,
    build_round_lineup,
    leverage_count_for_round,
)
from megax.probability import build_score_matrix_from_match
from megax.utility import compute_match_analysis


def _money(home: float = 70.0) -> dict:
    return {
        "tipsport": {"home": home, "draw": 15, "away": 15, "over": 50, "under": 50},
        "fortuna": {"home": home + 5, "draw": 8, "away": 10, "over": 55, "under": 45},
        "sazkabet": {"home": home + 3, "draw": 10, "away": 12, "over": 52, "under": 48},
    }


def _context(match, *, kickoff: datetime, home_money: float = 70.0) -> MatchLineupContext:
    prob = build_score_matrix_from_match(match)
    assert prob is not None
    crowd = build_crowd_matrix(prob, _money(home_money))
    analysis = compute_match_analysis(prob, crowd, field_size=50_000)
    return MatchLineupContext(
        match_id=match.match_id,
        kickoff_at=kickoff,
        analysis=analysis,
    )


def test_leverage_count_for_eight_match_round() -> None:
    assert leverage_count_for_round(8) == 2


def test_build_round_lineup_assigns_jokers_and_mixed_picks() -> None:
    from test_probability import _plzen_match

    base = datetime(2026, 7, 25, 15, 0, tzinfo=timezone.utc)
    plzen = _plzen_match()
    teplice = replace(
        plzen,
        match_id=8212281,
        name="Baník - Teplice",
        kickoff_at=base + timedelta(hours=26),
    )
    contexts = (
        _context(plzen, kickoff=base, home_money=85),
        _context(teplice, kickoff=base + timedelta(hours=26), home_money=55),
    )

    lineup = build_round_lineup(contexts, leverage_count=2)
    assert lineup.account_a.joker_match_id == plzen.match_id
    assert lineup.account_b.joker_match_id == teplice.match_id
    assert lineup.account_a.leverage_count >= 1
    assert lineup.account_b.leverage_count >= 1
    assert lineup.account_a.total_ev > 0


def test_accounts_avoid_identical_tips_when_possible() -> None:
    from test_probability import _plzen_match

    base = datetime(2026, 7, 25, 15, 0, tzinfo=timezone.utc)
    plzen = _plzen_match()
    teplice = replace(
        plzen,
        match_id=8212281,
        name="Baník - Teplice",
        kickoff_at=base + timedelta(hours=26),
    )
    contexts = (
        _context(plzen, kickoff=base),
        _context(teplice, kickoff=base + timedelta(hours=26)),
    )
    lineup = build_round_lineup(contexts)
    same = sum(
        1
        for pick_a, pick_b in zip(lineup.account_a.picks, lineup.account_b.picks, strict=True)
        if pick_a.tip == pick_b.tip
    )
    assert same < len(contexts)


def test_apply_lineup_to_state() -> None:
    from test_probability import _plzen_match

    plzen = _plzen_match()
    kickoff = datetime(2026, 7, 25, 15, 0, tzinfo=timezone.utc)
    contexts = (_context(plzen, kickoff=kickoff),)
    lineup = build_round_lineup(contexts, leverage_count=1)
    state = RoundGuiState()
    apply_lineup_to_state(state, lineup)
    match_id = str(plzen.match_id)
    assert state.accounts["A"].tips[match_id]
    assert state.accounts["B"].tips[match_id]
    assert state.accounts["A"].joker_match_id == plzen.match_id
