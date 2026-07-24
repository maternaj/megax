"""Fetch and structure a Megatipovačka round from Tipsport."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from megax.config import MegaxConfig, load_config
from megax.tipsport.client import TipsportClient
from megax.tipsport.offer import (
    KickoffSlot,
    MegaxMatch,
    fetch_competition_matches,
    filter_matches_by_kickoff_window,
    group_by_kickoff_slot,
)


@dataclass(frozen=True)
class RoundSnapshot:
    competition_id: int
    date_from: datetime
    date_to: datetime
    fetched_at: datetime
    matches: tuple[MegaxMatch, ...]
    slots: tuple[KickoffSlot, ...]


def fetch_round_snapshot(
    *,
    date_from: datetime,
    date_to: datetime,
    config: MegaxConfig | None = None,
    client: TipsportClient | None = None,
) -> RoundSnapshot:
    config = config or load_config()
    client = client or TipsportClient(
        config.tipsport_base_url,
        state_file=config.tipsport_state_file,
    )
    all_matches = fetch_competition_matches(
        client,
        competition_id=config.tipsport_competition_id,
    )
    window_matches = filter_matches_by_kickoff_window(
        all_matches,
        date_from=date_from,
        date_to=date_to,
    )
    if len(window_matches) > config.max_matches_per_round:
        window_matches = window_matches[: config.max_matches_per_round]
    slots = group_by_kickoff_slot(window_matches)
    return RoundSnapshot(
        competition_id=config.tipsport_competition_id,
        date_from=date_from.astimezone(timezone.utc),
        date_to=date_to.astimezone(timezone.utc),
        fetched_at=datetime.now(timezone.utc),
        matches=tuple(window_matches),
        slots=tuple(slots),
    )
