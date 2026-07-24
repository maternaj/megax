"""Late-swap state machine — protect/chase remaining picks by slot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from megax.config import MegaxConfig, load_config
from megax.lineup import RoundLineup, build_round_lineup, leverage_count_for_round
from megax.scoring import points
from megax.simulate import build_lineup_contexts
from megax.tipsport.offer import MegaxMatch
from megax.tipsport.results import MatchResult, MatchStatus

if TYPE_CHECKING:
    from megax.gui.state import RoundGuiState
    from megax.ingest import RoundSnapshot
    from megax.lineup import MatchLineupContext


class SwapMode(str, Enum):
    PROTECT = "protect"
    NEUTRAL = "neutral"
    CHASE = "chase"


@dataclass(frozen=True)
class SwapConfig:
    delta_small: float = 3.0
    delta_large: float = 8.0
    leader_chalk: float = 0.85
    chase_alpha_boost: float = 0.3
    protect_ev_ratio: float = 0.95
    chase_ev_ratio: float = 0.85


@dataclass(frozen=True)
class SwapChange:
    account: str
    match_id: int
    match_name: str
    current_tip: str
    recommended_tip: str
    pick_type: str


@dataclass(frozen=True)
class SwapRecommendation:
    mode: SwapMode
    delta: float
    leader_estimate: float
    our_points_a: int
    our_points_b: int
    our_best: int
    remaining_match_count: int
    remaining_match_ids: tuple[int, ...]
    next_slot_at: datetime | None
    lineup: RoundLineup
    changes: tuple[SwapChange, ...]


def swap_config_from_env(config: MegaxConfig | None = None) -> SwapConfig:
    cfg = config or load_config()
    return SwapConfig(
        delta_small=cfg.swap_delta_small,
        delta_large=cfg.swap_delta_large,
        leader_chalk=cfg.swap_leader_chalk,
        chase_alpha_boost=cfg.swap_chase_alpha_boost,
        protect_ev_ratio=cfg.swap_protect_ev_ratio,
        chase_ev_ratio=cfg.swap_chase_ev_ratio,
    )


def classify_swap_mode(delta: float, config: SwapConfig) -> SwapMode:
    if delta <= config.delta_small:
        return SwapMode.PROTECT
    if delta >= config.delta_large:
        return SwapMode.CHASE
    return SwapMode.NEUTRAL


def estimate_leader_points(contexts: tuple[MatchLineupContext, ...], config: SwapConfig) -> float:
    return sum(ctx.analysis.ev.best.ev for ctx in contexts) * config.leader_chalk


def _leverage_for_mode(mode: SwapMode, match_count: int) -> int:
    if match_count <= 0:
        return 0
    if mode == SwapMode.PROTECT:
        return 0
    if mode == SwapMode.CHASE:
        return max(2, min(3, round(match_count * 0.4)))
    return leverage_count_for_round(match_count)


def _mode_analysis_params(mode: SwapMode, config: SwapConfig) -> tuple[float | None, float]:
    if mode == SwapMode.PROTECT:
        return config.protect_ev_ratio, 0.0
    if mode == SwapMode.CHASE:
        return config.chase_ev_ratio, config.chase_alpha_boost
    return None, 0.0


def build_swap_lineup(
    remaining_matches: tuple[MegaxMatch, ...],
    state: RoundGuiState,
    mode: SwapMode,
    *,
    config: SwapConfig | None = None,
    megax_config: MegaxConfig | None = None,
) -> RoundLineup | None:
    if not remaining_matches:
        return None
    swap_cfg = config or swap_config_from_env(megax_config)
    gpp_ev_ratio, alpha_boost = _mode_analysis_params(mode, swap_cfg)
    contexts = build_lineup_contexts(
        remaining_matches,
        state,
        config=megax_config,
        gpp_ev_ratio=gpp_ev_ratio,
        alpha_boost=alpha_boost,
    )
    if len(contexts) != len(remaining_matches):
        return None
    leverage = _leverage_for_mode(mode, len(contexts))
    return build_round_lineup(contexts, leverage_count=leverage)


def _score_account(
    account: str,
    state: RoundGuiState,
    matches: tuple[MegaxMatch, ...],
    results: dict[int, MatchResult | None],
) -> int:
    total = 0
    account_state = state.accounts[account]
    for match in matches:
        result = results.get(match.match_id)
        if result is None or result.status != MatchStatus.FINISHED:
            continue
        if result.home_goals is None or result.away_goals is None:
            continue
        tip_text = account_state.tips.get(str(match.match_id), "")
        if not tip_text:
            continue
        parts = tip_text.split(":")
        if len(parts) != 2:
            continue
        try:
            tip_home = int(parts[0])
            tip_away = int(parts[1])
        except ValueError:
            continue
        scored = points(tip_home, tip_away, result.home_goals, result.away_goals)
        if account_state.joker_match_id == match.match_id:
            scored *= 2
        total += scored
    return total


def _remaining_matches(
    snapshot: RoundSnapshot,
    *,
    now: datetime,
) -> tuple[tuple[MegaxMatch, ...], datetime | None]:
    remaining: list[MegaxMatch] = []
    next_slot: datetime | None = None
    for slot in snapshot.slots:
        if slot.kickoff_at <= now:
            continue
        if next_slot is None:
            next_slot = slot.kickoff_at
        remaining.extend(slot.matches)
    return tuple(remaining), next_slot


def _collect_changes(
    *,
    state: RoundGuiState,
    lineup: RoundLineup,
    remaining_ids: set[int],
    matches_by_id: dict[int, MegaxMatch],
) -> tuple[SwapChange, ...]:
    changes: list[SwapChange] = []
    for account, account_lineup in (("A", lineup.account_a), ("B", lineup.account_b)):
        for pick in account_lineup.picks:
            if pick.match_id not in remaining_ids:
                continue
            current = state.accounts[account].tips.get(str(pick.match_id), "")
            if current == pick.tip:
                continue
            match = matches_by_id.get(pick.match_id)
            name = match.name if match else str(pick.match_id)
            changes.append(
                SwapChange(
                    account=account,
                    match_id=pick.match_id,
                    match_name=name,
                    current_tip=current or "—",
                    recommended_tip=pick.tip,
                    pick_type=pick.pick_type,
                )
            )
    return tuple(changes)


def compute_swap_recommendation(
    *,
    snapshot: RoundSnapshot,
    state: RoundGuiState,
    contexts: tuple[MatchLineupContext, ...],
    results: dict[int, MatchResult | None],
    now: datetime | None = None,
    config: SwapConfig | None = None,
    megax_config: MegaxConfig | None = None,
) -> SwapRecommendation | None:
    if not contexts or len(contexts) != len(snapshot.matches):
        return None

    swap_cfg = config or swap_config_from_env(megax_config)
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    remaining_matches, next_slot = _remaining_matches(snapshot, now=current)
    if not remaining_matches:
        return None

    our_points_a = _score_account("A", state, snapshot.matches, results)
    our_points_b = _score_account("B", state, snapshot.matches, results)
    chance_a = state.accounts["A"].points or 0
    chance_b = state.accounts["B"].points or 0
    our_best = max(our_points_a, our_points_b, chance_a, chance_b)

    leader_estimate = estimate_leader_points(contexts, swap_cfg)
    delta = leader_estimate - our_best
    mode = classify_swap_mode(delta, swap_cfg)

    lineup = build_swap_lineup(
        remaining_matches,
        state,
        mode,
        config=swap_cfg,
        megax_config=megax_config,
    )
    if lineup is None:
        return None

    remaining_ids = {match.match_id for match in remaining_matches}
    matches_by_id = {match.match_id: match for match in snapshot.matches}
    changes = _collect_changes(
        state=state,
        lineup=lineup,
        remaining_ids=remaining_ids,
        matches_by_id=matches_by_id,
    )

    return SwapRecommendation(
        mode=mode,
        delta=delta,
        leader_estimate=leader_estimate,
        our_points_a=max(our_points_a, chance_a),
        our_points_b=max(our_points_b, chance_b),
        our_best=our_best,
        remaining_match_count=len(remaining_matches),
        remaining_match_ids=tuple(sorted(remaining_ids)),
        next_slot_at=next_slot,
        lineup=lineup,
        changes=changes,
    )


def apply_swap_to_state(
    state: RoundGuiState,
    recommendation: SwapRecommendation,
    *,
    remaining_match_ids: set[int],
) -> None:
    """Apply recommended tips for remaining (unstarted) matches only."""
    for account, account_lineup in (("A", recommendation.lineup.account_a), ("B", recommendation.lineup.account_b)):
        for pick in account_lineup.picks:
            if pick.match_id not in remaining_match_ids:
                continue
            state.accounts[account].tips[str(pick.match_id)] = pick.tip
