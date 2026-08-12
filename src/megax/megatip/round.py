"""Round id detection and resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from megax.megatip.api import MegatipApi, clients_tips_path, resolve_round_id
from megax.megatip.parse import round_number_from_entry


@dataclass(frozen=True)
class ActualRound:
    round_id: int
    round_number: int | None
    status: str | None = None


def actual_round_from_round_list(
    round_list: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> ActualRound | None:
    """Return the round marked actualRound in clients/tips roundList."""
    for raw in round_list:
        if not isinstance(raw, dict):
            continue
        if not raw.get("actualRound"):
            continue
        round_id = raw.get("roundId")
        if round_id is None:
            continue
        status_raw = raw.get("roundStatus")
        status = status_raw.get("statusType") if isinstance(status_raw, dict) else None
        return ActualRound(
            round_id=int(round_id),
            round_number=round_number_from_entry(raw),
            status=status,
        )
    return None


def detect_current_round_id(
    api: MegatipApi,
    *,
    offset: int = 380,
) -> int | None:
    """Auto-detect active roundId from tile and roundList."""
    tile = api.fetch_tile(actual=True)
    if tile is not None and tile.current_round is not None:
        resolved = resolve_round_id(api, tile.current_round, offset=offset)
        if resolved is not None:
            return resolved

    data = api.transport.fetch(clients_tips_path(api.contest_id), auth=False)
    if isinstance(data, dict):
        actual = actual_round_from_round_list(data.get("roundList", []))
        if actual is not None:
            return actual.round_id
    return None
