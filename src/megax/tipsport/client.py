"""Tipsport REST client — thin wrapper over shared bookmaker session."""

from __future__ import annotations

from typing import Any

import cloudscraper25

from megax.bookmaker.client import BookmakerClient
from megax.bookmaker.session import (
    ScraperState,
    create_scraper,
    default_state_file,
    exponential_backoff,
    init_web_request,
    load_scraper_state,
    request_json,
    save_scraper_state,
)


def load_saved_scraper(state_file: str) -> cloudscraper25.CloudScraper | None:
    loaded = load_scraper_state(state_file)
    if loaded is None:
        return None
    scraper, _state = loaded
    return scraper


def save_successful_scraper(scraper: cloudscraper25.CloudScraper, state_file: str) -> bool:
    return save_scraper_state(scraper, state_file)


def fetch_json(
    scraper: cloudscraper25.CloudScraper,
    base_url: str,
    endpoint: str,
) -> tuple[dict[str, Any] | None, int | None]:
    return request_json(scraper, base_url, endpoint, method="GET")


class TipsportClient(BookmakerClient):
    """Self-contained Tipsport REST client (init-web session + cloudscraper)."""

    def fetch_match_results(self, match_id: int) -> dict[str, Any] | None:
        return self.fetch(
            f"/rest/offer/v3/matches/{match_id}?fromResults=true",
            retry=False,
        )


__all__ = [
    "ScraperState",
    "TipsportClient",
    "create_scraper",
    "default_state_file",
    "exponential_backoff",
    "fetch_json",
    "init_web_request",
    "load_saved_scraper",
    "save_successful_scraper",
]
