"""Parse Tipsport match results API responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from megax.tipsport.client import TipsportClient

DEFAULT_RESULTS_POLL_DELAY = timedelta(hours=1, minutes=45)


def should_poll_match_results(
    kickoff_at: datetime,
    *,
    now: datetime | None = None,
    min_delay: timedelta = DEFAULT_RESULTS_POLL_DELAY,
) -> bool:
    """Return True once enough time has passed after kickoff to expect results."""
    now = now or datetime.now(timezone.utc)
    if kickoff_at.tzinfo is None:
        kickoff_at = kickoff_at.replace(tzinfo=timezone.utc)
    else:
        kickoff_at = kickoff_at.astimezone(timezone.utc)
    return now >= kickoff_at + min_delay


class MatchStatus(str, Enum):
    PENDING = "pending"
    LIVE = "live"
    FINISHED = "finished"


@dataclass(frozen=True)
class ResultCell:
    opp_id: int
    odd: float | None
    winning: bool | None
    observed_at: datetime | None


@dataclass(frozen=True)
class MatchResult:
    match_id: int
    status: MatchStatus
    home_goals: int | None
    away_goals: int | None
    ended: bool
    observed_at: datetime | None


def _parse_date_closed(raw: object) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ms_to_utc(raw: object) -> datetime | None:
    if raw is None:
        return None
    try:
        ms = int(raw)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def match_has_results(match_data: dict[str, Any]) -> bool:
    match = match_data.get("match") or {}
    result_parts = match.get("resultParts")
    return bool(result_parts)


def parse_ft_score(match_data: dict[str, Any]) -> tuple[int, int] | None:
    """Return full-time goals (home, away) from resultParts."""
    match = match_data.get("match") or {}
    parts = match.get("resultParts")
    if not parts or len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        return None


def parse_result_cells(match_data: dict[str, Any]) -> dict[int, ResultCell]:
    """Map opp_id -> result cell from fromResults=true payload."""
    if not match_has_results(match_data):
        return {}

    cells: dict[int, ResultCell] = {}
    match = match_data.get("match") or {}
    for event_table in match.get("eventTables") or []:
        for box in event_table.get("boxes") or []:
            for cell in box.get("cells") or []:
                if "winning" not in cell:
                    continue
                opp_id_raw = cell.get("id")
                if opp_id_raw is None:
                    continue
                opp_id = int(opp_id_raw)
                winning_raw = cell.get("winning")
                winning = winning_raw if isinstance(winning_raw, bool) else None
                odd_raw = cell.get("odd")
                odd = float(odd_raw) if odd_raw is not None else None
                cells[opp_id] = ResultCell(
                    opp_id=opp_id,
                    odd=odd,
                    winning=winning,
                    observed_at=_parse_date_closed(cell.get("dateClosed")),
                )
    return cells


def _latest_observed_at(cells: dict[int, ResultCell]) -> datetime | None:
    observed = [cell.observed_at for cell in cells.values() if cell.observed_at is not None]
    if not observed:
        return None
    return max(observed)


def infer_match_status(
    match_data: dict[str, Any],
    *,
    now: datetime | None = None,
) -> MatchStatus:
    now = now or datetime.now(timezone.utc)
    if match_has_results(match_data):
        return MatchStatus.FINISHED

    match = match_data.get("match") or {}
    if bool(match.get("ended")):
        return MatchStatus.FINISHED

    kickoff_at = _ms_to_utc(match.get("dateStart"))
    if kickoff_at is not None and kickoff_at <= now:
        return MatchStatus.LIVE
    return MatchStatus.PENDING


def parse_match_result(match_data: dict[str, Any]) -> MatchResult | None:
    match = match_data.get("match") or {}
    match_id_raw = match.get("id")
    if match_id_raw is None:
        return None

    score = parse_ft_score(match_data)
    status = infer_match_status(match_data)
    cells = parse_result_cells(match_data)
    return MatchResult(
        match_id=int(match_id_raw),
        status=status,
        home_goals=score[0] if score else None,
        away_goals=score[1] if score else None,
        ended=bool(match.get("ended")) or status == MatchStatus.FINISHED,
        observed_at=_latest_observed_at(cells),
    )


def fetch_match_result(client: TipsportClient, match_id: int) -> MatchResult | None:
    payload = client.fetch_match_results(match_id)
    if payload is None:
        return None
    return parse_match_result(payload)


def poll_match_results(
    client: TipsportClient,
    match_ids: list[int],
    *,
    kickoffs: dict[int, datetime] | None = None,
    min_delay: timedelta | None = None,
) -> dict[int, MatchResult | None]:
    results: dict[int, MatchResult | None] = {}
    now = datetime.now(timezone.utc)
    delay = min_delay or DEFAULT_RESULTS_POLL_DELAY
    for match_id in match_ids:
        kickoff = (kickoffs or {}).get(match_id)
        if kickoff is not None and not should_poll_match_results(kickoff, now=now, min_delay=delay):
            results[match_id] = None
            continue
        results[match_id] = fetch_match_result(client, match_id)
    return results
