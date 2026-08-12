"""Megatipovačka REST endpoint builders and high-level client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from megax.bookmaker.client import BookmakerClient
from megax.megatip.parse import (
    parse_joker_response,
    parse_popular_tip_probe,
    parse_ranking,
    parse_round_tips,
    parse_tile,
)
from megax.megatip.models import (
    JokerActionResult,
    PopularTipProbe,
    RankingSnapshot,
    RoundTipsSnapshot,
    TileSnapshot,
)


def clients_tips_path(contest_id: int, *, round_id: int | None = None) -> str:
    query = f"?roundId={round_id}" if round_id is not None else "?"
    return f"/rest/contests/v1/megatipovacka/{contest_id}/clients/tips{query}"


def popular_tips_path(round_match_id: int, home: int, away: int) -> str:
    return (
        "/rest/megatipovacka/v1/competitions/round-matches/"
        f"{round_match_id}/popular-tips"
        f"?firstParticipantTip={home}&secondParticipantTip={away}"
    )


def ranking_path(
    contest_id: int,
    *,
    serie_id: int | None = None,
    round_id: int | None = None,
    limit: int = 50,
    page: int = 1,
    actual: bool = False,
) -> str:
    params = [f"limit={limit}", f"page={page}", f"actual={'true' if actual else 'false'}"]
    if serie_id is not None:
        params.insert(0, f"serieId={serie_id}")
    if round_id is not None:
        params.insert(1 if serie_id is not None else 0, f"roundId={round_id}")
    return f"/rest/megatipovacka/v1/competitions/{contest_id}/ranking?{'&'.join(params)}"


def tile_path(tile_id: int, *, actual: bool = True) -> str:
    actual_flag = "true" if actual else "false"
    return f"/rest/contests/v1/tiles/MEGATIPOVACKA/{tile_id}?actual={actual_flag}"


def submit_tip_path(contest_id: int, round_match_id: int) -> str:
    return (
        f"/rest/megatipovacka/v1/competitions/{contest_id}/round-matches/"
        f"{round_match_id}/client-tips"
    )


def joker_path(contest_id: int, round_match_id: int) -> str:
    return (
        f"/rest/megatipovacka/v1/competitions/{contest_id}/round-matches/"
        f"{round_match_id}/joker"
    )


class MegatipTransport(Protocol):
    def fetch(self, endpoint: str, *, retry: bool = True, auth: bool = False) -> dict[str, Any] | None: ...

    def put(
        self,
        endpoint: str,
        body: dict[str, Any] | None = None,
        *,
        retry: bool = True,
        auth: bool = True,
    ) -> dict[str, Any] | None: ...

    def delete(
        self,
        endpoint: str,
        *,
        retry: bool = True,
        auth: bool = True,
    ) -> dict[str, Any] | None: ...


@dataclass
class MegatipApi:
    transport: MegatipTransport
    contest_id: int
    tile_id: int
    serie_id: int

    def fetch_round_tips(
        self,
        round_id: int | None = None,
        *,
        auth: bool = False,
    ) -> RoundTipsSnapshot | None:
        data = self.transport.fetch(
            clients_tips_path(self.contest_id, round_id=round_id),
            auth=auth,
        )
        if data is None:
            return None
        return parse_round_tips(data, contest_id=self.contest_id, round_id=round_id)

    def fetch_popular_tip(
        self,
        round_match_id: int,
        home: int,
        away: int,
    ) -> PopularTipProbe | None:
        data = self.transport.fetch(
            popular_tips_path(round_match_id, home, away),
            auth=True,
        )
        if data is None:
            return None
        return parse_popular_tip_probe(
            data,
            round_match_id=round_match_id,
            home=home,
            away=away,
        )

    def fetch_ranking(
        self,
        *,
        round_id: int | None = None,
        serie_id: int | None = None,
        limit: int = 50,
        page: int = 1,
        actual: bool = False,
    ) -> RankingSnapshot | None:
        resolved_serie = self.serie_id if serie_id is None else serie_id
        data = self.transport.fetch(
            ranking_path(
                self.contest_id,
                serie_id=resolved_serie,
                round_id=round_id,
                limit=limit,
                page=page,
                actual=actual,
            ),
            auth=True,
        )
        if data is None:
            return None
        return parse_ranking(
            data,
            contest_id=self.contest_id,
            serie_id=resolved_serie,
            round_id=round_id,
        )

    def fetch_tile(self, *, actual: bool = True, auth: bool = False) -> TileSnapshot | None:
        data = self.transport.fetch(tile_path(self.tile_id, actual=actual), auth=auth)
        if data is None:
            return None
        return parse_tile(data, tile_id=self.tile_id)

    def submit_tip(self, round_match_id: int, home: int, away: int) -> dict[str, Any] | None:
        return self.transport.put(
            submit_tip_path(self.contest_id, round_match_id),
            {"firstOpponentTip": home, "secondOpponentTip": away},
            auth=True,
        )

    def assign_joker(self, round_match_id: int) -> JokerActionResult | None:
        data = self.transport.put(joker_path(self.contest_id, round_match_id), auth=True)
        if data is None:
            return None
        return parse_joker_response(data)

    def remove_joker(self, round_match_id: int) -> JokerActionResult | None:
        data = self.transport.delete(joker_path(self.contest_id, round_match_id), auth=True)
        if data is None:
            return None
        return parse_joker_response(data)


def megatip_api_from_client(
    client: BookmakerClient,
    *,
    contest_id: int,
    tile_id: int,
    serie_id: int,
) -> MegatipApi:
    return MegatipApi(
        transport=client,
        contest_id=contest_id,
        tile_id=tile_id,
        serie_id=serie_id,
    )


def resolve_round_id(
    api: MegatipApi,
    round_number: int,
    *,
    offset: int = 380,
) -> int:
    """Resolve roundId from roundList, falling back to offset+round_number."""
    from megax.megatip.parse import megatip_round_id, round_id_from_round_list

    data = api.transport.fetch(
        clients_tips_path(api.contest_id),
        auth=False,
    )
    if data is not None:
        resolved = round_id_from_round_list(data.get("roundList", []), round_number)
        if resolved is not None:
            return resolved
    return megatip_round_id(round_number, offset=offset)
