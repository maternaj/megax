"""Submit Megatipovačka lineups and joker via REST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from megax.megatip.api import MegatipApi
from megax.megatip.errors import LineupSubmitError, MegatipJokerError
from megax.megatip.models import JokerActionResult, RoundTipsSnapshot


@dataclass(frozen=True)
class LineupSubmission:
    round_match_id: int
    home: int
    away: int


def submit_lineup(
    api: MegatipApi,
    submissions: list[LineupSubmission],
) -> list[tuple[LineupSubmission, dict[str, Any] | None]]:
    results: list[tuple[LineupSubmission, dict[str, Any] | None]] = []
    for item in submissions:
        response = api.submit_tip(item.round_match_id, item.home, item.away)
        results.append((item, response))
    return results


def lineup_from_match_tips(
    snapshot: RoundTipsSnapshot,
    tips_by_match_id: dict[int, tuple[int, int]],
    *,
    strict: bool = True,
) -> list[LineupSubmission]:
    by_match = snapshot.by_match_id()
    submissions: list[LineupSubmission] = []
    missing: list[int] = []
    for match_id, (home, away) in tips_by_match_id.items():
        match = by_match.get(match_id)
        if match is None:
            missing.append(match_id)
            continue
        submissions.append(
            LineupSubmission(
                round_match_id=match.round_match_id,
                home=home,
                away=away,
            )
        )
    if strict and missing:
        raise LineupSubmitError(missing)
    return submissions


def submit_lineup_for_matches(
    api: MegatipApi,
    snapshot: RoundTipsSnapshot,
    tips_by_match_id: dict[int, tuple[int, int]],
) -> list[tuple[LineupSubmission, dict[str, Any] | None]]:
    return submit_lineup(api, lineup_from_match_tips(snapshot, tips_by_match_id))


def current_joker_round_match_id(snapshot: RoundTipsSnapshot) -> int | None:
    for tip in snapshot.client_tips:
        if tip.joker_used:
            return tip.round_match_id
    return None


def set_joker(
    api: MegatipApi,
    round_match_id: int,
    *,
    current_joker_round_match_id: int | None = None,
) -> JokerActionResult:
    """Assign joker to a match, optionally moving it from another match first."""
    if current_joker_round_match_id == round_match_id:
        return JokerActionResult(
            joker_used=True,
            free_joker=False,
            message="Joker already on this match",
            refresh_tip_page=False,
        )
    if (
        current_joker_round_match_id is not None
        and current_joker_round_match_id != round_match_id
    ):
        removed = api.remove_joker(current_joker_round_match_id)
        if removed is None:
            raise MegatipJokerError("Failed to remove joker from current match")
        if not removed.free_joker:
            raise MegatipJokerError(removed.message or "Could not free joker slot")
    assigned = api.assign_joker(round_match_id)
    if assigned is None:
        raise MegatipJokerError("Joker assign request failed")
    if not assigned.joker_used:
        raise MegatipJokerError(assigned.message or "Joker was not applied")
    return assigned


def set_joker_for_match(
    api: MegatipApi,
    snapshot: RoundTipsSnapshot,
    joker_match_id: int,
) -> JokerActionResult:
    match = snapshot.by_match_id().get(joker_match_id)
    if match is None:
        raise LineupSubmitError([joker_match_id])
    return set_joker(
        api,
        match.round_match_id,
        current_joker_round_match_id=current_joker_round_match_id(snapshot),
    )
