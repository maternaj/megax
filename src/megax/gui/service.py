"""Build GUI view models from Tipsport ingest + manual state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from megax.crowd import CrowdMatrixResult
from megax.crowd_observed import CROWD_GRID_SIZE, build_crowd_matrix_from_observed
from megax.gui.state import RoundGuiState
from megax.ingest import RoundSnapshot, fetch_round_snapshot
from megax.poll import poll_once
from megax.probability import ScoreMatrixResult, build_score_matrix_from_match
from megax.scoring import points
from megax.storage import RoundRecord, load_round_record, save_round_record
from megax.tipsport.client import TipsportClient
from megax.tipsport.offer import MegaxMatch, group_by_kickoff_slot
from megax.lineup import MatchLineupContext, RoundLineup
from megax.swap import SwapRecommendation
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
    round_id: int | None = None


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
    round_id: int | None = None,
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
        pollable_ids = [
            match.match_id for match in snapshot.matches if match.odds.home > 0
        ]
        if pollable_ids:
            poll = poll_once(
                pollable_ids,
                kickoffs={mid: kickoffs[mid] for mid in pollable_ids},
                client=client,
            )
            results = poll.results
            results_polled_at = poll.polled_at
        else:
            results = {}
            results_polled_at = None
    elif results is None and snapshot.matches and read_only and client is not None:
        kickoffs = _match_kickoffs(snapshot.matches)
        pollable_ids = [
            match.match_id for match in snapshot.matches if match.odds.home > 0
        ]
        if pollable_ids:
            poll = poll_once(
                pollable_ids,
                kickoffs={mid: kickoffs[mid] for mid in pollable_ids},
                client=client,
            )
            results = poll.results
            results_polled_at = poll.polled_at
        else:
            results = {}
            results_polled_at = None
    else:
        results_polled_at = datetime.now(timezone.utc) if results else None

    totals_a = 0
    totals_b = 0
    finished_count = 0
    slot_views: list[SlotView] = []
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
            state.seed_crowd_from_megatip(match.match_id)
            crowd = (
                build_crowd_matrix_from_observed(
                    state.crowd_cells_for_match(match.match_id),
                    grid_size=CROWD_GRID_SIZE,
                    prob=prob,
                    top3_keys=state.top3_cell_keys(match.match_id),
                )
                if prob
                else None
            )

            analysis = None
            if prob and crowd and (
                (crowd.known and any(any(row) for row in crowd.known))
                or (crowd.estimated and any(any(row) for row in crowd.estimated))
            ):
                try:
                    analysis = compute_match_analysis(
                        prob,
                        crowd,
                        field_size=state.field_size,
                    )
                except ValueError:
                    analysis = None
            if analysis is not None:
                lineup_contexts.append(
                    MatchLineupContext(
                        match_id=match.match_id,
                        kickoff_at=match.kickoff_at,
                        analysis=analysis,
                    )
                )
            else:
                pass

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
        lineup=None,
        swap=None,
        read_only=read_only,
        saved_at=saved_at,
        round_id=round_id or state.round_id,
    )
