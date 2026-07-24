"""Round-level lineup optimizer — chalk/leverage mix for two accounts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from megax.ev import TipCandidate
from megax.utility import (
    DEFAULT_FIELD_SIZE,
    MatchTipAnalysis,
    UtilityCandidate,
    utility_score,
)

if TYPE_CHECKING:
    from megax.gui.state import RoundGuiState


@dataclass(frozen=True)
class MatchLineupContext:
    match_id: int
    kickoff_at: datetime
    analysis: MatchTipAnalysis


@dataclass(frozen=True)
class MatchPick:
    match_id: int
    tip: str
    pick_type: str
    ev: float
    crowd_share: float
    utility: float | None = None

    @property
    def is_leverage(self) -> bool:
        return self.pick_type == "leverage"


@dataclass(frozen=True)
class AccountLineup:
    account: str
    picks: tuple[MatchPick, ...]
    joker_match_id: int
    total_ev: float
    leverage_count: int

    def tips_by_match(self) -> dict[int, str]:
        return {pick.match_id: pick.tip for pick in self.picks}


@dataclass(frozen=True)
class RoundLineup:
    account_a: AccountLineup
    account_b: AccountLineup
    leverage_match_ids: tuple[int, ...]


def leverage_count_for_round(match_count: int) -> int:
    if match_count <= 0:
        return 0
    return max(2, min(3, round(match_count * 0.3)))


def joker_value(analysis: MatchTipAnalysis) -> float:
    """Score for joker placement: double EV on the GPP tip, scaled by crowd."""
    tip = analysis.gpp_best
    return utility_score(tip.ev * 2.0, tip.crowd_share, alpha=analysis.gpp_alpha)


def _crowd_share_for_tip(analysis: MatchTipAnalysis, tip: str) -> float:
    for candidate in analysis.gpp_top:
        if candidate.score == tip:
            return candidate.crowd_share
    return analysis.gpp_best.crowd_share


def _alternate_tip(
    analysis: MatchTipAnalysis,
    *,
    avoid: set[str],
    prefer_leverage: bool,
) -> TipCandidate | UtilityCandidate:
    pools: list[tuple[TipCandidate | UtilityCandidate, ...]] = []
    if prefer_leverage:
        pools.append(analysis.gpp_top)
    pools.append(analysis.ev.top)
    if not prefer_leverage:
        pools.append(analysis.gpp_top)
    for pool in pools:
        for candidate in pool:
            if candidate.score not in avoid:
                return candidate
    return analysis.ev.best


def _pick_from_candidate(
    ctx: MatchLineupContext,
    candidate: TipCandidate | UtilityCandidate,
    *,
    pick_type: str,
) -> MatchPick:
    if isinstance(candidate, UtilityCandidate):
        return MatchPick(
            match_id=ctx.match_id,
            tip=candidate.score,
            pick_type=pick_type,
            ev=candidate.ev,
            crowd_share=candidate.crowd_share,
            utility=candidate.utility if pick_type == "leverage" else None,
        )
    return MatchPick(
        match_id=ctx.match_id,
        tip=candidate.score,
        pick_type=pick_type,
        ev=candidate.ev,
        crowd_share=_crowd_share_for_tip(ctx.analysis, candidate.score),
        utility=None,
    )


def _assign_leverage_matches(
    contexts: tuple[MatchLineupContext, ...],
    *,
    leverage_count: int,
) -> tuple[frozenset[int], frozenset[int]]:
    if leverage_count <= 0 or not contexts:
        return frozenset(), frozenset()

    ranked = sorted(
        contexts,
        key=lambda ctx: (-ctx.analysis.gpp_best.utility, ctx.kickoff_at, ctx.match_id),
    )
    a_count = (leverage_count + 1) // 2
    b_count = leverage_count - a_count

    a_ids: set[int] = set()
    b_ids: set[int] = set()

    for ctx in ranked:
        if len(a_ids) >= a_count:
            break
        a_ids.add(ctx.match_id)

    for ctx in reversed(ranked):
        if ctx.match_id in a_ids:
            continue
        if len(b_ids) >= b_count:
            break
        b_ids.add(ctx.match_id)

    if len(b_ids) < b_count:
        for ctx in ranked:
            if ctx.match_id in a_ids or ctx.match_id in b_ids:
                continue
            b_ids.add(ctx.match_id)
            if len(b_ids) >= b_count:
                break

    return frozenset(a_ids), frozenset(b_ids)


def _pick_joker(
    contexts: tuple[MatchLineupContext, ...],
    *,
    earliest: bool,
) -> int:
    if not contexts:
        raise ValueError("Cannot pick joker without matches")
    target_kickoff = (
        min(ctx.kickoff_at for ctx in contexts)
        if earliest
        else max(ctx.kickoff_at for ctx in contexts)
    )
    slot = [ctx for ctx in contexts if ctx.kickoff_at == target_kickoff]
    best = max(slot, key=lambda ctx: (joker_value(ctx.analysis), ctx.analysis.gpp_best.ev))
    return best.match_id


def _build_account_lineup(
    contexts: tuple[MatchLineupContext, ...],
    *,
    account: str,
    leverage_ids: frozenset[int],
    other_tips: dict[int, str],
) -> AccountLineup:
    picks: list[MatchPick] = []
    for ctx in contexts:
        avoid = {other_tips.get(ctx.match_id, "")} - {""}
        if ctx.match_id in leverage_ids:
            candidate = _alternate_tip(ctx.analysis, avoid=avoid, prefer_leverage=True)
            pick = _pick_from_candidate(ctx, candidate, pick_type="leverage")
        else:
            candidate = _alternate_tip(ctx.analysis, avoid=avoid, prefer_leverage=False)
            pick = _pick_from_candidate(ctx, candidate, pick_type="chalk")
        picks.append(pick)

    joker = _pick_joker(contexts, earliest=(account == "A"))
    return AccountLineup(
        account=account,
        picks=tuple(picks),
        joker_match_id=joker,
        total_ev=sum(pick.ev for pick in picks),
        leverage_count=sum(1 for pick in picks if pick.is_leverage),
    )


def build_round_lineup(
    contexts: tuple[MatchLineupContext, ...],
    *,
    leverage_count: int | None = None,
) -> RoundLineup:
    if not contexts:
        raise ValueError("Cannot build lineup without matches")

    count = leverage_count if leverage_count is not None else leverage_count_for_round(len(contexts))
    a_leverage, b_leverage = _assign_leverage_matches(contexts, leverage_count=count)

    account_a = _build_account_lineup(
        contexts,
        account="A",
        leverage_ids=a_leverage,
        other_tips={},
    )
    account_b = _build_account_lineup(
        contexts,
        account="B",
        leverage_ids=b_leverage,
        other_tips=account_a.tips_by_match(),
    )

    all_leverage = tuple(sorted(set(a_leverage) | set(b_leverage)))
    return RoundLineup(
        account_a=account_a,
        account_b=account_b,
        leverage_match_ids=all_leverage,
    )


def apply_lineup_to_state(state: RoundGuiState, lineup: RoundLineup) -> None:
    """Write optimizer tips and jokers into RoundGuiState."""
    for pick in lineup.account_a.picks:
        state.accounts["A"].tips[str(pick.match_id)] = pick.tip
    state.accounts["A"].joker_match_id = lineup.account_a.joker_match_id

    for pick in lineup.account_b.picks:
        state.accounts["B"].tips[str(pick.match_id)] = pick.tip
    state.accounts["B"].joker_match_id = lineup.account_b.joker_match_id
