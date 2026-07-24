"""Tipsport REST integration for MegaX."""

from megax.tipsport.client import TipsportClient
from megax.tipsport.offer import (
    MegaxMatch,
    MatchOdds,
    competition_offer_endpoint,
    fetch_competition_matches,
    group_by_kickoff_slot,
    parse_match,
)
from megax.tipsport.results import MatchResult, MatchStatus, parse_ft_score, parse_match_result

__all__ = [
    "competition_offer_endpoint",
    "MatchOdds",
    "MatchResult",
    "MatchStatus",
    "MegaxMatch",
    "TipsportClient",
    "fetch_competition_matches",
    "group_by_kickoff_slot",
    "parse_ft_score",
    "parse_match",
    "parse_match_result",
]
