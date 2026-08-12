"""Client login for Tipsport/Chance authenticated REST APIs."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import cloudscraper25

from megax.bookmaker.session import request_json

logger = logging.getLogger(__name__)

SESSION_PATH = "/rest/client/v5/session"
DEFAULT_MEGATIP_REFERER_PATH = "/souteze/detail/megatipovacka/3575/tipy"


def login_request(
    scraper: cloudscraper25.CloudScraper,
    base_url: str,
    username: str,
    password: str,
    *,
    device_id: str | None = None,
    referer: str | None = None,
) -> tuple[dict[str, Any] | None, int | None, str]:
    """POST /rest/client/v5/session. Returns (body, status, device_id)."""
    resolved_device_id = device_id or str(uuid.uuid4())
    payload = {
        "username": username,
        "password": password,
        "consents": {
            "targeting": True,
            "performance": True,
            "timestamp": int(time.time() * 1000),
        },
        "autoLogin": False,
        "deviceId": resolved_device_id,
    }
    resolved_referer = referer or f"{base_url.rstrip('/')}{DEFAULT_MEGATIP_REFERER_PATH}"
    scraper.headers["Referer"] = resolved_referer
    result, status = request_json(
        scraper,
        base_url,
        SESSION_PATH,
        method="POST",
        json_body=payload,
    )
    if result is not None:
        logger.info("Login succeeded for %s", username)
    else:
        body_hint = ""
        if status == 418:
            body_hint = " (anti-bot/WAF — browser jatvgptmow headers required; seed state/chance_scraper_state.json from a logged-in browser session)"
        logger.warning(
            "Login failed for %s (status=%s)%s. If this persists, seed state/%s from a "
            "browser session or check anti-bot / Cloudflare blocks.",
            username,
            status,
            body_hint,
            "chance_scraper_state.json",
        )
    return result, status, resolved_device_id
