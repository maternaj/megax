"""Parse Tipsport bulk offer for Czech league Megatipovačka rounds."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from megax.team_mu import TeamOuLine
from megax.tipsport.client import TipsportClient

WINNER_3W = "16-WINNER_3W-1"
ASIAN_TOTAL_FT = "16-ASIAN_TOTAL-1"
TEAM_TOTAL_FT = "16-TOTAL_PARTICIPANT-1"

COMPETITION_OFFER_TEMPLATE = (
    "/rest/external/offer/v1/matches?idCompetition={competition_id}&allEvents=true"
)

_LINE_RE = re.compile(r"(?:Více|Méně) než ([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class MatchOdds:
    home: float
    draw: float
    away: float
    over_2_5: float | None = None
    under_2_5: float | None = None
    home_team_lines: tuple[TeamOuLine, ...] = ()
    away_team_lines: tuple[TeamOuLine, ...] = ()
    match_total_lines: tuple[TeamOuLine, ...] = ()


@dataclass(frozen=True)
class MegaxMatch:
    match_id: int
    name: str
    home: str
    away: str
    kickoff_at: datetime
    odds: MatchOdds
    match_type: str
    ended: bool
    competition_id: int


@dataclass(frozen=True)
class KickoffSlot:
    kickoff_at: datetime
    matches: tuple[MegaxMatch, ...]


def competition_offer_endpoint(competition_id: int) -> str:
    return COMPETITION_OFFER_TEMPLATE.format(competition_id=competition_id)


def _ms_to_utc(raw: object) -> datetime | None:
    if raw is None:
        return None
    try:
        ms = int(raw)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _parse_winner_3w(event: dict[str, Any]) -> tuple[float, float, float] | None:
    home = draw = away = None
    for opp in event.get("opps") or []:
        if not opp.get("bettingEnabled", True):
            continue
        odd_raw = opp.get("odd")
        if odd_raw is None:
            continue
        odd = float(odd_raw)
        side = str(opp.get("type") or "").lower()
        if side == "1":
            home = odd
        elif side in {"x", "0"}:
            draw = odd
        elif side == "2":
            away = odd
    if home is None or draw is None or away is None:
        return None
    return home, draw, away


def _line_from_opp_name(name: str) -> float | None:
    match = _LINE_RE.search(name or "")
    if not match:
        return None
    return float(match.group(1))


def _is_quarter_line(name: str) -> bool:
    return "(" in (name or "")


def _pick_total_25_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for event in events:
        if event.get("mySelectionId") != ASIAN_TOTAL_FT:
            continue
        opps = [o for o in (event.get("opps") or []) if o.get("bettingEnabled", True)]
        if len(opps) != 2:
            continue
        lines = {_line_from_opp_name(str(o.get("name") or "")) for o in opps}
        lines.discard(None)
        if not lines:
            continue
        line = next(iter(lines))
        quarter = any(_is_quarter_line(str(o.get("name") or "")) for o in opps)
        if line == 2.5 and not quarter:
            score = 0
        elif line == 2.5 and quarter:
            score = 1
        elif line == 2.25:
            score = 2
        else:
            continue
        candidates.append((score, event))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _parse_over_under(event: dict[str, Any]) -> tuple[float, float] | None:
    over = under = None
    for opp in event.get("opps") or []:
        if not opp.get("bettingEnabled", True):
            continue
        odd_raw = opp.get("odd")
        if odd_raw is None:
            continue
        odd = float(odd_raw)
        name = str(opp.get("name") or "")
        side = str(opp.get("type") or "").lower()
        if name.startswith("Více") or side == "o":
            over = odd
        elif name.startswith("Méně") or side == "u":
            under = odd
    if over is None or under is None:
        return None
    return over, under


def _parse_match_total_lines(events: list[dict[str, Any]]) -> tuple[TeamOuLine, ...]:
    lines: dict[float, TeamOuLine] = {}
    for event in events:
        if event.get("mySelectionId") != ASIAN_TOTAL_FT:
            continue
        title = str(event.get("name") or "").lower()
        if "poločasu" in title:
            continue
        opps = [o for o in (event.get("opps") or []) if o.get("bettingEnabled", True)]
        if len(opps) != 2:
            continue
        if any(_is_quarter_line(str(o.get("name") or "")) for o in opps):
            continue
        line = _line_from_opp_name(str(opps[0].get("name") or ""))
        if line is None:
            continue
        totals = _parse_over_under(event)
        if totals is None:
            continue
        over, under = totals
        lines[line] = TeamOuLine(line=line, over=over, under=under)
    return tuple(lines[line] for line in sorted(lines))


def _parse_team_total_lines(
    events: list[dict[str, Any]],
    *,
    home_name: str,
    away_name: str,
) -> tuple[tuple[TeamOuLine, ...], tuple[TeamOuLine, ...]]:
    home_lines: dict[float, TeamOuLine] = {}
    away_lines: dict[float, TeamOuLine] = {}

    for event in events:
        if event.get("mySelectionId") != TEAM_TOTAL_FT:
            continue
        title = str(event.get("name") or "")
        if "poločasu" in title.lower():
            continue
        if home_name and home_name in title:
            bucket = home_lines
        elif away_name and away_name in title:
            bucket = away_lines
        else:
            continue

        opps = [o for o in (event.get("opps") or []) if o.get("bettingEnabled", True)]
        if len(opps) != 2:
            continue
        if any(_is_quarter_line(str(o.get("name") or "")) for o in opps):
            continue
        line = _line_from_opp_name(str(opps[0].get("name") or ""))
        if line is None:
            continue
        totals = _parse_over_under(event)
        if totals is None:
            continue
        over, under = totals
        bucket[line] = TeamOuLine(line=line, over=over, under=under)

    return (
        tuple(home_lines[line] for line in sorted(home_lines)),
        tuple(away_lines[line] for line in sorted(away_lines)),
    )


def _iter_market_events(raw: dict[str, Any]) -> list[dict[str, Any]]:
    events = list(raw.get("events") or [])
    main = raw.get("mainEvent")
    if isinstance(main, dict):
        events.append(main)
    return events


def parse_match(raw: dict[str, Any]) -> MegaxMatch | None:
    kickoff_at = _ms_to_utc(raw.get("dateStart"))
    if kickoff_at is None:
        return None

    events = _iter_market_events(raw)
    winner_event = next((ev for ev in events if ev.get("mySelectionId") == WINNER_3W), None)
    if winner_event is None:
        main = raw.get("mainEvent")
        if isinstance(main, dict) and main.get("mySelectionId") == WINNER_3W:
            winner_event = main
    if winner_event is None:
        return None

    winner = _parse_winner_3w(winner_event)
    if winner is None:
        return None
    home_odd, draw_odd, away_odd = winner
    home_name = str(raw.get("homeParticipant") or "")
    away_name = str(raw.get("visitingParticipant") or "")

    total_event = _pick_total_25_event(events)
    over_2_5 = under_2_5 = None
    if total_event is not None:
        totals = _parse_over_under(total_event)
        if totals is not None:
            over_2_5, under_2_5 = totals

    home_team_lines, away_team_lines = _parse_team_total_lines(
        events,
        home_name=home_name,
        away_name=away_name,
    )
    match_total_lines = _parse_match_total_lines(events)

    return MegaxMatch(
        match_id=int(raw["id"]),
        name=str(raw.get("name") or raw.get("nameFull") or ""),
        home=home_name,
        away=away_name,
        kickoff_at=kickoff_at,
        odds=MatchOdds(
            home=home_odd,
            draw=draw_odd,
            away=away_odd,
            over_2_5=over_2_5,
            under_2_5=under_2_5,
            home_team_lines=home_team_lines,
            away_team_lines=away_team_lines,
            match_total_lines=match_total_lines,
        ),
        match_type=str(raw.get("matchType") or ""),
        ended=bool(raw.get("ended")),
        competition_id=int(raw.get("idCompetition") or 0),
    )


def fetch_competition_matches(
    client: TipsportClient,
    *,
    competition_id: int,
) -> list[MegaxMatch]:
    endpoint = competition_offer_endpoint(competition_id)
    payload = client.fetch(endpoint)
    if payload is None:
        return []
    matches: list[MegaxMatch] = []
    for raw in payload.get("matches") or []:
        if int(raw.get("idCompetition") or 0) != competition_id:
            continue
        parsed = parse_match(raw)
        if parsed is not None:
            matches.append(parsed)
    matches.sort(key=lambda m: (m.kickoff_at, m.match_id))
    return matches


def filter_matches_by_kickoff_window(
    matches: list[MegaxMatch],
    *,
    date_from: datetime,
    date_to: datetime,
) -> list[MegaxMatch]:
    if date_from.tzinfo is None:
        date_from = date_from.replace(tzinfo=timezone.utc)
    else:
        date_from = date_from.astimezone(timezone.utc)
    if date_to.tzinfo is None:
        date_to = date_to.replace(tzinfo=timezone.utc)
    else:
        date_to = date_to.astimezone(timezone.utc)
    return [m for m in matches if date_from <= m.kickoff_at <= date_to]


def group_by_kickoff_slot(matches: list[MegaxMatch]) -> list[KickoffSlot]:
    if not matches:
        return []
    ordered = sorted(matches, key=lambda m: (m.kickoff_at, m.match_id))
    slots: list[KickoffSlot] = []
    current_kickoff = ordered[0].kickoff_at
    bucket: list[MegaxMatch] = []
    for match in ordered:
        if match.kickoff_at != current_kickoff:
            slots.append(KickoffSlot(kickoff_at=current_kickoff, matches=tuple(bucket)))
            current_kickoff = match.kickoff_at
            bucket = [match]
        else:
            bucket.append(match)
    if bucket:
        slots.append(KickoffSlot(kickoff_at=current_kickoff, matches=tuple(bucket)))
    return slots
