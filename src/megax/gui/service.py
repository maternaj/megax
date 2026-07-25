"""Build GUI view models from Tipsport ingest + manual state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from megax.config import load_config
from megax.crowd import CrowdMatrixResult, build_crowd_matrix
from megax.gui.state import RoundGuiState, parse_tip_score
from megax.ingest import RoundSnapshot, fetch_round_snapshot
from megax.poll import poll_once
from megax.probability import ScoreMatrixResult, build_score_matrix_from_match
from megax.scoring import points
from megax.storage import RoundRecord, load_round_record, save_round_record
from megax.tipsport.client import TipsportClient
from megax.tipsport.offer import MegaxMatch, group_by_kickoff_slot
from megax.calibrate import build_lineup_for_knobs, knobs_from_snapshot
from megax.lineup import MatchLineupContext, RoundLineup, build_round_lineup
from megax.swap import SwapRecommendation, compute_swap_recommendation
from megax.tipsport.results import MatchResult, MatchStatus
from megax.utility import MatchTipAnalysis, compute_match_analysis


@dataclass(frozen=True)
class MatchRow:
    match: MegaxMatch
    result: MatchResult | None
    points_a: int | None
    points_b: int | None
    probability: ScoreMatrixResult | None
    crowd: CrowdMatrixResult | None
    analysis: MatchTipAnalysis | None


@dataclass(frozen=True)
class SlotView:
    kickoff_at: datetime
    matches: tuple[MatchRow, ...]


@dataclass(frozen=True)
class RoundView:
    snapshot: RoundSnapshot
    round_key: str
    state: RoundGuiState
    slots: tuple[SlotView, ...]
    fetched_at: datetime
    results_polled_at: datetime | None
    totals_a: int
    totals_b: int
    finished_count: int
    lineup: RoundLineup | None = None
    swap: SwapRecommendation | None = None
    read_only: bool = False
    saved_at: datetime | None = None


def _score_tip(
    tip_text: str,
    result: MatchResult | None,
    *,
    joker: bool,
) -> int | None:
    if result is None or result.home_goals is None or result.away_goals is None:
        return None
    tip = parse_tip_score(tip_text)
    if tip is None:
        return None
    base = points(tip[0], tip[1], result.home_goals, result.away_goals)
    return base * 2 if joker else base


def snapshot_from_record(
    record: RoundRecord,
    *,
    date_from: datetime,
    date_to: datetime,
) -> RoundSnapshot:
    matches = list(record.matches)
    return RoundSnapshot(
        competition_id=matches[0].competition_id if matches else 120,
        date_from=date_from.astimezone(timezone.utc),
        date_to=date_to.astimezone(timezone.utc),
        fetched_at=record.fetched_at or record.saved_at,
        matches=tuple(matches),
        slots=tuple(group_by_kickoff_slot(matches)),
    )


def _match_kickoffs(matches: tuple[MegaxMatch, ...]) -> dict[int, datetime]:
    return {match.match_id: match.kickoff_at for match in matches}


def build_round_view(
    *,
    date_from: datetime,
    date_to: datetime,
    round_key: str,
    state: RoundGuiState,
    snapshot: RoundSnapshot | None = None,
    results: dict[int, MatchResult | None] | None = None,
    client: TipsportClient | None = None,
    read_only: bool = False,
    saved_at: datetime | None = None,
) -> RoundView:
    if snapshot is None:
        snapshot = fetch_round_snapshot(
            date_from=date_from,
            date_to=date_to,
            client=client,
        )
    for match in snapshot.matches:
        state.ensure_match(match.match_id)

    if results is None and snapshot.matches and not read_only:
        kickoffs = _match_kickoffs(snapshot.matches)
        poll = poll_once(
            [match.match_id for match in snapshot.matches],
            kickoffs=kickoffs,
            client=client,
        )
        results = poll.results
        results_polled_at = poll.polled_at
    elif results is None and snapshot.matches and read_only and client is not None:
        kickoffs = _match_kickoffs(snapshot.matches)
        poll = poll_once(
            [match.match_id for match in snapshot.matches],
            kickoffs=kickoffs,
            client=client,
        )
        results = poll.results
        results_polled_at = poll.polled_at
    else:
        results_polled_at = datetime.now(timezone.utc) if results else None

    totals_a = 0
    totals_b = 0
    finished_count = 0
    slot_views: list[SlotView] = []
    config = load_config()
    crowd_blend_to_p = config.crowd_blend_to_p
    crowd_tail_gamma = config.crowd_tail_gamma
    crowd_zero_zero_delta = config.crowd_zero_zero_delta
    crowd_prelec_alpha = config.crowd_prelec_alpha
    crowd_zero_zero_min = config.crowd_zero_zero_min
    gpp_alpha = config.gpp_alpha

    lineup_contexts: list[MatchLineupContext] = []

    for slot in snapshot.slots:
        rows: list[MatchRow] = []
        for match in slot.matches:
            result = (results or {}).get(match.match_id)
            if result and result.status == MatchStatus.FINISHED:
                finished_count += 1
            account_a = state.accounts["A"]
            account_b = state.accounts["B"]
            tip_a = account_a.tips.get(str(match.match_id), "")
            tip_b = account_b.tips.get(str(match.match_id), "")
            joker_a = account_a.joker_match_id == match.match_id
            joker_b = account_b.joker_match_id == match.match_id
            pts_a = _score_tip(tip_a, result, joker=joker_a)
            pts_b = _score_tip(tip_b, result, joker=joker_b)
            if pts_a is not None:
                totals_a += pts_a
            if pts_b is not None:
                totals_b += pts_b

            prob = build_score_matrix_from_match(match)
            crowd = (
                build_crowd_matrix(
                    prob,
                    state.money[str(match.match_id)],
                    blend_to_p=crowd_blend_to_p,
                    tail_gamma=crowd_tail_gamma,
                    zero_zero_delta=crowd_zero_zero_delta,
                    prelec_alpha=crowd_prelec_alpha,
                    zero_zero_min=crowd_zero_zero_min,
                )
                if prob
                else None
            )

            if prob and crowd:
                analysis = compute_match_analysis(
                    prob,
                    crowd,
                    field_size=state.field_size,
                    gpp_alpha=gpp_alpha,
                )
                lineup_contexts.append(
                    MatchLineupContext(
                        match_id=match.match_id,
                        kickoff_at=match.kickoff_at,
                        analysis=analysis,
                    )
                )
            else:
                analysis = None

            rows.append(
                MatchRow(
                    match=match,
                    result=result,
                    points_a=pts_a,
                    points_b=pts_b,
                    probability=prob,
                    crowd=crowd,
                    analysis=analysis,
                )
            )
        slot_views.append(SlotView(kickoff_at=slot.kickoff_at, matches=tuple(rows)))

    lineup = None
    if lineup_contexts and len(lineup_contexts) == len(snapshot.matches):
        if state.calibration is not None:
            lineup = build_lineup_for_knobs(
                snapshot.matches,
                state,
                knobs_from_snapshot(state.calibration),
                config=config,
            )
        if lineup is None:
            lineup = build_round_lineup(tuple(lineup_contexts))

    swap = None
    if not read_only and lineup_contexts and len(lineup_contexts) == len(snapshot.matches):
        swap = compute_swap_recommendation(
            snapshot=snapshot,
            state=state,
            contexts=tuple(lineup_contexts),
            results=results or {},
            megax_config=config,
        )

    return RoundView(
        snapshot=snapshot,
        round_key=round_key,
        state=state,
        slots=tuple(slot_views),
        fetched_at=snapshot.fetched_at,
        results_polled_at=results_polled_at,
        totals_a=totals_a,
        totals_b=totals_b,
        finished_count=finished_count,
        lineup=lineup,
        swap=swap,
        read_only=read_only,
        saved_at=saved_at,
    )
