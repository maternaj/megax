"""Parse Megatipovačka REST JSON payloads."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from megax.megatip.models import (
    ClientTip,
    JokerActionResult,
    PopularTipProbe,
    PopularTips,
    RankingEntry,
    RankingSnapshot,
    RoundMatch,
    RoundTipsSnapshot,
    ScoreTip,
    TileSnapshot,
)

_INT_RE = re.compile(r"[^0-9]+")


def parse_score_label(value: str) -> tuple[int, int] | None:
    if not value or ":" not in value:
        return None
    left, right = value.split(":", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def _parse_tip(raw: dict[str, Any] | None) -> ScoreTip | None:
    if not raw:
        return None
    parsed = parse_score_label(str(raw.get("value", "")))
    if parsed is None:
        return None
    home, away = parsed
    percentage = int(raw.get("percentage", 0))
    return ScoreTip(home=home, away=away, percentage=percentage, is_floor=percentage <= 1)


def parse_popular_tips(raw: dict[str, Any] | None) -> PopularTips | None:
    if not raw:
        return None
    first = _parse_tip(raw.get("firstPopularTip"))
    second = _parse_tip(raw.get("secondPopularTip"))
    third = _parse_tip(raw.get("thirdPopularTip"))
    if first is None or second is None or third is None:
        return None
    return PopularTips(
        message=raw.get("message"),
        top3=(first, second, third),
    )


def _parse_iso_datetime(raw: object) -> datetime | None:
    if not raw:
        return None
    text = str(raw)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _parse_result(raw: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not raw:
        return None, None
    final = parse_score_label(str(raw.get("resultFinal", "")))
    if final is None:
        try:
            return int(raw.get("home")), int(raw.get("away"))
        except (TypeError, ValueError):
            return None, None
    return final


def parse_round_match(raw: dict[str, Any]) -> RoundMatch | None:
    round_match_id = raw.get("roundMatchId")
    match_id = raw.get("matchId")
    if round_match_id is None or match_id is None:
        return None
    result_home, result_away = _parse_result(raw.get("result"))
    return RoundMatch(
        round_match_id=int(round_match_id),
        match_id=int(match_id),
        match_name=str(raw.get("matchName", "")),
        kickoff_at=_parse_iso_datetime(raw.get("matchDate")),
        status=str(raw.get("status", "")),
        popular_tips=parse_popular_tips(raw.get("popularTips")),
        result_home=result_home,
        result_away=result_away,
    )


def parse_client_tips(raw: dict[str, Any] | None) -> tuple[ClientTip, ...]:
    if not raw:
        return ()
    tips_block = raw.get("clientTips")
    if not isinstance(tips_block, list):
        nested = raw.get("tips")
        if isinstance(nested, dict):
            client = nested.get("client")
            if isinstance(client, dict) and isinstance(client.get("tips"), list):
                tips_block = client["tips"]
            elif isinstance(nested.get("tips"), list):
                tips_block = nested["tips"]
        elif isinstance(nested, list):
            tips_block = nested
    if not isinstance(tips_block, list):
        return ()
    parsed: list[ClientTip] = []
    for item in tips_block:
        if not isinstance(item, dict):
            continue
        round_match_id = item.get("roundMatchId")
        if round_match_id is None:
            continue
        home = item.get("firstParticipantTip", item.get("firstOpponentTip"))
        away = item.get("secondParticipantTip", item.get("secondOpponentTip"))
        if home is None or away is None:
            continue
        parsed.append(
            ClientTip(
                round_match_id=int(round_match_id),
                home=int(home),
                away=int(away),
                score=int(item["score"]) if item.get("score") is not None else None,
                joker_used=bool(item.get("jokerUsed", False)),
            )
        )
    return tuple(parsed)


def parse_round_tips(data: dict[str, Any], *, contest_id: int, round_id: int | None) -> RoundTipsSnapshot:
    matches: list[RoundMatch] = []
    for raw in data.get("roundMatches", []):
        if not isinstance(raw, dict):
            continue
        parsed = parse_round_match(raw)
        if parsed is not None:
            matches.append(parsed)
    client_tips = parse_client_tips(data)
    return RoundTipsSnapshot(
        contest_id=contest_id,
        round_id=round_id,
        round_matches=tuple(matches),
        client_tips=client_tips,
        can_tip=bool(data.get("canTip", True)),
    )


def parse_popular_tip_probe(
    data: dict[str, Any],
    *,
    round_match_id: int,
    home: int,
    away: int,
) -> PopularTipProbe | None:
    popular = parse_popular_tips(data)
    if popular is None:
        return None
    client = _parse_tip(data.get("clientTip"))
    queried = client or ScoreTip(home=home, away=away, percentage=0)
    for tip in popular.top3:
        if tip.home == home and tip.away == away:
            queried = tip
            client = None
            break
    return PopularTipProbe(
        round_match_id=round_match_id,
        queried=queried,
        popular_tips=popular,
        client_tip=client,
    )


def _parse_ranking_entry(raw: dict[str, Any], *, is_client: bool = False) -> RankingEntry | None:
    position = raw.get("position")
    score = raw.get("score")
    if position is None or score is None:
        return None
    username = None
    client = raw.get("client")
    if isinstance(client, dict):
        avatar = client.get("avatar")
        if isinstance(avatar, dict):
            username = avatar.get("username")
    return RankingEntry(
        position=int(position),
        score=int(score),
        username=username,
        prize=raw.get("prize"),
        is_client=is_client,
    )


def parse_ranking(
    data: dict[str, Any],
    *,
    contest_id: int,
    serie_id: int | None,
    round_id: int | None,
) -> RankingSnapshot:
    entries: list[RankingEntry] = []
    for raw in data.get("clientsRankingEntries", []):
        if not isinstance(raw, dict):
            continue
        parsed = _parse_ranking_entry(raw)
        if parsed is not None:
            entries.append(parsed)
    client_entry = None
    raw_client = data.get("clientRankingEntry")
    if isinstance(raw_client, dict):
        client_entry = _parse_ranking_entry(raw_client, is_client=True)
    leader_score = entries[0].score if entries else None
    return RankingSnapshot(
        contest_id=contest_id,
        serie_id=serie_id,
        round_id=round_id,
        number_of_players=int(data.get("numberOfPlayersTotal", 0)),
        entries=tuple(entries),
        client_entry=client_entry,
        leader_score=leader_score,
    )


def _parse_int_from_text(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    digits = _INT_RE.sub("", str(value))
    if not digits:
        return None
    return int(digits)


def parse_tile(data: dict[str, Any], *, tile_id: int) -> TileSnapshot:
    rank = field_size = points = current_round = total_rounds = None
    next_kickoff_text = None
    for row in data.get("rows", []):
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", ""))
        values = row.get("values") or []
        if "pořadí" in title.lower() and len(values) >= 2:
            rank = _parse_int_from_text(values[0].get("value"))
            field_size = _parse_int_from_text(values[1].get("value"))
        elif "počet bodů" in title.lower() and values:
            points = _parse_int_from_text(values[0].get("value"))
        elif "začátek" in title.lower() and values:
            next_kickoff_text = str(values[0].get("value"))
        elif "kolo" in title.lower() and "celkov" in title.lower() and len(values) >= 2:
            current_round = _parse_int_from_text(values[0].get("value"))
            total_rounds = _parse_int_from_text(values[1].get("value"))
    return TileSnapshot(
        contest_id=int(data.get("contestId", data.get("tipovackaId", 0))),
        tile_id=tile_id,
        rank=rank,
        field_size=field_size,
        points=points,
        current_round=current_round,
        total_rounds=total_rounds,
        next_kickoff_text=next_kickoff_text,
    )


def parse_joker_response(data: dict[str, Any]) -> JokerActionResult:
    return JokerActionResult(
        joker_used=bool(data.get("jokerUsed", False)),
        free_joker=bool(data.get("freeJoker", False)),
        message=data.get("jokerSnackbarText"),
        refresh_tip_page=bool(data.get("refreshTipPage", False)),
    )


_ROUND_NAME_RE = re.compile(r"^(\d+)\.")


def round_number_from_entry(raw: dict[str, Any]) -> int | None:
    if raw.get("roundNumber") is not None:
        return int(raw["roundNumber"])
    name = str(raw.get("roundName", ""))
    match = _ROUND_NAME_RE.match(name)
    if match:
        return int(match.group(1))
    return None


def round_id_from_round_list(
    round_list: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    round_number: int,
) -> int | None:
    """Resolve API roundId from clients/tips or ranking roundList."""
    for raw in round_list:
        if not isinstance(raw, dict):
            continue
        round_id = raw.get("roundId")
        if round_id is None:
            continue
        if raw.get("roundNumber") == round_number:
            return int(round_id)
        name = str(raw.get("roundName", ""))
        match = _ROUND_NAME_RE.match(name)
        if match and int(match.group(1)) == round_number:
            return int(round_id)
    return None


def megatip_round_id(round_number: int, *, offset: int = 380) -> int:
    """Fallback map 1-based round number to roundId when roundList is unavailable."""
    return offset + round_number
