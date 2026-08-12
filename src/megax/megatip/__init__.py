"""Megatipovačka REST integration."""

from megax.megatip.api import MegatipApi, megatip_api_from_client
from megax.megatip.crowd_probe import build_observed_coverage
from megax.megatip.models import (
    CrowdObservedSnapshot,
    JokerActionResult,
    PopularTipProbe,
    RankingSnapshot,
    RoundTipsSnapshot,
    TileSnapshot,
)
from megax.megatip.parse import megatip_round_id
from megax.megatip.round import ActualRound, actual_round_from_round_list, detect_current_round_id
from megax.megatip.submit import (
    LineupSubmission,
    set_joker,
    set_joker_for_match,
    submit_lineup,
    submit_lineup_for_matches,
)

__all__ = [
    "CrowdObservedSnapshot",
    "JokerActionResult",
    "LineupSubmission",
    "MegatipApi",
    "PopularTipProbe",
    "RankingSnapshot",
    "RoundTipsSnapshot",
    "TileSnapshot",
    "build_observed_coverage",
    "megatip_api_from_client",
    "ActualRound",
    "actual_round_from_round_list",
    "detect_current_round_id",
    "megatip_round_id",
    "set_joker",
    "set_joker_for_match",
    "submit_lineup",
    "submit_lineup_for_matches",
]
