"""Bridge Megatip round data with Tipsport odds for the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time, timezone

from megax.config import MegaxConfig, load_config
from megax.gui.state import MegatipGuiCache, MegatipMatchGui, RoundGuiState
from megax.crowd_observed import merge_api_top3_into_cells
from megax.ingest import RoundSnapshot
from megax.megatip.api import MegatipApi, megatip_api_from_client
from megax.megatip.models import RoundMatch, RoundTipsSnapshot
from megax.megatip.round import actual_round_from_round_list
from megax.tipsport.client import TipsportClient
from megax.tipsport.offer import MegaxMatch, MatchOdds, fetch_competition_matches, group_by_kickoff_slot


@dataclass(frozen=True)
class MegatipFetchResult:
    round_id: int
    round_number: int | None
    round_tips: RoundTipsSnapshot
    snapshot: RoundSnapshot
    missing_match_ids: tuple[int, ...]


def megatip_api(config: MegaxConfig | None = None) -> MegatipApi:
    """Public Megatip reads via anonymous Tipsport session (no login)."""
    config = config or load_config()
    client = TipsportClient(
        config.tipsport_base_url,
        state_file=config.tipsport_state_file,
    )
    return megatip_api_from_client(
        client,
        contest_id=config.megatip_contest_id,
        tile_id=config.megatip_tile_id,
        serie_id=config.megatip_serie_id,
    )


def _day_bounds_from_matches(matches: tuple[MegaxMatch, ...]) -> tuple[datetime, datetime]:
    if not matches:
        now = datetime.now(timezone.utc)
        return now, now
    kickoffs = [match.kickoff_at for match in matches]
    start_day = min(kickoffs).astimezone(timezone.utc).date()
    end_day = max(kickoffs).astimezone(timezone.utc).date()
    date_from = datetime.combine(start_day, dt_time.min, tzinfo=timezone.utc)
    date_to = datetime.combine(end_day, dt_time.max, tzinfo=timezone.utc)
    return date_from, date_to


def _tipsport_matches_for_round(
    round_matches: tuple[RoundMatch, ...],
    *,
    config: MegaxConfig,
    client: TipsportClient | None,
) -> tuple[dict[int, MegaxMatch], tuple[int, ...]]:
    client = client or TipsportClient(
        config.tipsport_base_url,
        state_file=config.tipsport_state_file,
    )
    all_matches = fetch_competition_matches(
        client,
        competition_id=config.tipsport_competition_id,
    )
    by_id = {match.match_id: match for match in all_matches}
    wanted = {match.match_id for match in round_matches}
    found = {match_id: by_id[match_id] for match_id in wanted if match_id in by_id}
    missing = tuple(sorted(wanted - found.keys()))
    return found, missing


def _stub_megax_match(round_match: RoundMatch, *, competition_id: int) -> MegaxMatch:
    """Placeholder match when Tipsport comp offer has no odds yet."""
    kickoff = round_match.kickoff_at or datetime.now(timezone.utc)
    name = round_match.match_name
    if " - " in name:
        home, away = name.split(" - ", 1)
    else:
        home, away = name, ""
    return MegaxMatch(
        match_id=round_match.match_id,
        name=name,
        home=home.strip(),
        away=away.strip(),
        kickoff_at=kickoff,
        odds=MatchOdds(home=0.0, draw=0.0, away=0.0),
        match_type=round_match.status,
        ended=round_match.status != "MATCH_PREMATCH",
        competition_id=competition_id,
    )


def _ordered_tipsport_matches(
    round_matches: tuple[RoundMatch, ...],
    tipsport_by_id: dict[int, MegaxMatch],
    *,
    competition_id: int,
) -> tuple[MegaxMatch, ...]:
    ordered: list[MegaxMatch] = []
    for round_match in round_matches:
        tipsport = tipsport_by_id.get(round_match.match_id)
        if tipsport is not None:
            ordered.append(tipsport)
        else:
            ordered.append(_stub_megax_match(round_match, competition_id=competition_id))
    return tuple(ordered)


def fetch_megatip_round(
    round_id: int,
    *,
    config: MegaxConfig | None = None,
    tipsport_client: TipsportClient | None = None,
    megatip: MegatipApi | None = None,
) -> MegatipFetchResult | None:
    """Fetch public Megatip round tips (top-3 per match) and join Tipsport odds."""
    config = config or load_config()
    api = megatip or megatip_api(config)
    round_tips = api.fetch_round_tips(round_id, auth=False)
    if round_tips is None:
        return None

    round_number = _round_number_for_id(api, round_id)
    tipsport_by_id, missing = _tipsport_matches_for_round(
        round_tips.round_matches,
        config=config,
        client=tipsport_client,
    )
    matches = _ordered_tipsport_matches(
        round_tips.round_matches,
        tipsport_by_id,
        competition_id=config.tipsport_competition_id,
    )
    if len(matches) > config.max_matches_per_round:
        matches = matches[: config.max_matches_per_round]
    date_from, date_to = _day_bounds_from_matches(matches)
    snapshot = RoundSnapshot(
        competition_id=config.tipsport_competition_id,
        date_from=date_from,
        date_to=date_to,
        fetched_at=datetime.now(timezone.utc),
        matches=matches,
        slots=tuple(group_by_kickoff_slot(matches)),
    )
    return MegatipFetchResult(
        round_id=round_id,
        round_number=round_number,
        round_tips=round_tips,
        snapshot=snapshot,
        missing_match_ids=missing,
    )


def _round_number_for_id(api: MegatipApi, round_id: int) -> int | None:
    from megax.megatip.api import clients_tips_path
    from megax.megatip.parse import round_number_from_entry

    data = api.transport.fetch(clients_tips_path(api.contest_id), auth=False)
    if not isinstance(data, dict):
        return None
    for raw in data.get("roundList", []):
        if not isinstance(raw, dict):
            continue
        if raw.get("roundId") == round_id:
            return round_number_from_entry(raw)
    actual = actual_round_from_round_list(data.get("roundList", []))
    if actual is not None and actual.round_id == round_id:
        return actual.round_number
    return None


def apply_megatip_to_state(state: RoundGuiState, result: MegatipFetchResult) -> None:
    """Merge public Megatip top-3 data into GUI state."""
    state.round_id = result.round_id
    state.round_number = result.round_number

    matches_cache: dict[str, MegatipMatchGui] = {}
    for round_match in result.round_tips.round_matches:
        top3: dict[str, int] = {}
        if round_match.popular_tips is not None:
            top3 = round_match.popular_tips.as_dict()
        matches_cache[str(round_match.match_id)] = MegatipMatchGui(
            round_match_id=round_match.round_match_id,
            top3=top3,
            client_tip=None,
            status=round_match.status,
            result_home=round_match.result_home,
            result_away=round_match.result_away,
        )
        match_key = str(round_match.match_id)
        if top3:
            state.ensure_match(round_match.match_id)
            state.crowd_cells[match_key] = merge_api_top3_into_cells(
                state.crowd_cells.get(match_key, {}),
                top3,
                overwrite=False,
            )

    state.megatip = MegatipGuiCache(
        fetched_at=datetime.now(timezone.utc).isoformat(),
        round_number=result.round_number,
        field_size=None,
        client_rank=None,
        client_points=None,
        leader_score=None,
        can_tip=result.round_tips.can_tip,
        missing_match_ids=result.missing_match_ids,
        matches=matches_cache,
    )
