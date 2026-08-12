"""Shared Tipsport/Chance REST session and HTTP client."""

from megax.bookmaker.client import BookmakerClient
from megax.bookmaker.session import ScraperState, default_state_file

__all__ = [
    "BookmakerClient",
    "ScraperState",
    "default_state_file",
]
