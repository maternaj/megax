"""Cloudscraper session bootstrap, persistence, and low-level HTTP helpers."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import random
import re
import time
from datetime import datetime, timedelta
from typing import Any, Literal

import cloudscraper25
from fake_useragent import UserAgent
from user_agents import parse

logger = logging.getLogger(__name__)

INIT_WEB_PATH = "/rest/common/v1/init-web"
SCRAPER_MAX_AGE_HOURS = 24
HttpMethod = Literal["GET", "POST", "PUT", "DELETE"]


class ScraperState:
    def __init__(
        self,
        cookies: dict,
        headers: dict,
        user_agent: str,
        browser_info: dict,
        *,
        device_id: str | None = None,
        logged_in: bool = False,
    ):
        self.cookies = cookies
        self.headers = headers
        self.user_agent = user_agent
        self.browser_info = browser_info
        self.device_id = device_id
        self.logged_in = logged_in
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cookies": self.cookies,
            "headers": self.headers,
            "user_agent": self.user_agent,
            "browser_info": self.browser_info,
            "created_at": self.created_at,
            "logged_in": self.logged_in,
        }
        if self.device_id is not None:
            payload["device_id"] = self.device_id
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScraperState:
        return cls(
            cookies=data["cookies"],
            headers=data["headers"],
            user_agent=data["user_agent"],
            browser_info=data["browser_info"],
            device_id=data.get("device_id"),
            logged_in=bool(data.get("logged_in", False)),
        )


def default_state_file(name: str = "tipsport_scraper_state.json") -> str:
    root = pathlib.Path(__file__).resolve().parents[3]
    return str(root / "state" / name)


def get_clean_user_agent() -> str:
    ua = UserAgent(platforms="pc", min_version=122)
    while True:
        user_agent = ua.random
        if not re.search(
            r"Agency|Trailer|OpenWave|EdgiOS|Unique|AtContent|GLS|Config|EdgA|PTST|Viewer|Herring|Version",
            user_agent,
        ):
            return user_agent


def exponential_backoff(attempt: int) -> None:
    base_delay = min(60, 2 ** (attempt - 1))
    delay = min(60, base_delay + random.random())
    logger.info("Backing off for %.2f seconds (attempt %d)", delay, attempt)
    time.sleep(delay)


def generate_sec_ch_ua_header(browser: str, version: str) -> str:
    major = version.split(".")[0]
    if browser == "Chrome":
        return f'"Chromium";v="{major}", "Google Chrome";v="{major}", "Not-A.Brand";v="8"'
    if browser == "Firefox":
        return f'"Firefox";v="{major}", "Not-A.Brand";v="8"'
    if browser == "Safari":
        return f'"Safari";v="{major}", "Not-A.Brand";v="8"'
    return '"Not-A.Brand";v="8"'


def randomize_headers(headers: dict[str, str]) -> dict[str, str]:
    items = list(headers.items())
    random.shuffle(items)
    return dict(items)


def create_scraper() -> cloudscraper25.CloudScraper:
    return cloudscraper25.create_scraper(
        interpreter="js2py",
        enable_stealth=True,
        stealth_options={
            "min_delay": 2.0,
            "max_delay": 6.0,
            "human_like_delays": True,
            "randomize_headers": True,
            "browser_quirks": True,
        },
    )


def init_web_request(scraper: cloudscraper25.CloudScraper, base_url: str) -> bool:
    try:
        user_agent = get_clean_user_agent()
        ua = parse(user_agent)
        sec_ch_ua_header = generate_sec_ch_ua_header(ua.browser.family, ua.browser.version_string)
        session_headers = randomize_headers(
            {
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-type": "application/json;charset=utf-8",
                "DNT": "1",
                "Origin": base_url,
                "Platform": "WEB",
                "Pragma": "no-cache",
                "Priority": "u=1, i",
                "Referer": f"{base_url}/",
                "Sec-CH-UA": sec_ch_ua_header,
                "Sec-CH-UA-Mobile": "?1" if ua.is_mobile else "?0",
                "Sec-CH-UA-Platform": f'"{ua.os.family}"',
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "Sec-GPC": "1",
                "User-Agent": user_agent,
            }
        )
        scraper.headers.update(session_headers)
        payload = {
            "parameters": ["CLIENT_ANALYSIS__SHOW_LIVE_MATCHES"],
            "restartSession": False,
            "platform": "WEB",
            "consents": {},
        }
        url = f"{base_url.rstrip('/')}{INIT_WEB_PATH}"
        response = scraper.post(url, json=payload, headers=scraper.headers, timeout=30)
        logger.info("init-web status=%s", response.status_code)
        return response.status_code == 200
    except Exception:
        logger.exception("init-web failed")
        return False


def save_scraper_state(
    scraper: cloudscraper25.CloudScraper,
    state_file: str,
    *,
    device_id: str | None = None,
    logged_in: bool = False,
) -> bool:
    try:
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        cookies_dict = {}
        for cookie in scraper.cookies:
            cookies_dict[cookie.name] = {
                "value": cookie.value,
                "domain": cookie.domain,
                "path": cookie.path,
                "secure": cookie.secure,
                "expires": cookie.expires,
            }
        user_agent = scraper.headers.get("User-Agent", "")
        ua = parse(user_agent)
        state = ScraperState(
            cookies=cookies_dict,
            headers=dict(scraper.headers),
            user_agent=user_agent,
            browser_info={
                "browser": ua.browser.family,
                "version": ua.browser.version_string,
                "os": ua.os.family,
                "is_mobile": ua.is_mobile,
            },
            device_id=device_id,
            logged_in=logged_in,
        )
        with open(state_file, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2)
        return True
    except Exception:
        logger.exception("Failed to save scraper state")
        return False


def load_scraper_state(state_file: str) -> tuple[cloudscraper25.CloudScraper, ScraperState] | None:
    try:
        path = pathlib.Path(state_file)
        if not path.exists():
            return None
        file_age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if file_age > timedelta(hours=SCRAPER_MAX_AGE_HOURS):
            logger.info("Scraper state expired (%.1fh old)", file_age.total_seconds() / 3600)
            return None
        with path.open(encoding="utf-8") as fh:
            state = ScraperState.from_dict(json.load(fh))
        scraper = create_scraper()
        scraper.headers.clear()
        scraper.headers.update(state.headers)
        for name, cookie_data in state.cookies.items():
            scraper.cookies.set(
                name=name,
                value=cookie_data["value"],
                domain=cookie_data["domain"],
                path=cookie_data["path"],
                secure=cookie_data["secure"],
                expires=cookie_data["expires"],
            )
        return scraper, state
    except Exception:
        logger.exception("Failed to load scraper state")
        return None


def request_json(
    scraper: cloudscraper25.CloudScraper,
    base_url: str,
    endpoint: str,
    *,
    method: HttpMethod = "GET",
    json_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, int | None]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    headers = dict(scraper.headers)
    if method == "GET":
        response = scraper.get(url, headers=headers, timeout=30)
    elif method == "POST":
        response = scraper.post(url, json=json_body, headers=headers, timeout=30)
    elif method == "DELETE":
        response = scraper.delete(url, headers=headers, timeout=30)
    elif json_body is None:
        response = scraper.put(url, headers=headers, timeout=30)
    else:
        response = scraper.put(url, json=json_body, headers=headers, timeout=30)
    logger.info("%s %s -> %s", method, endpoint, response.status_code)
    if response.status_code not in (200, 201):
        return None, response.status_code
    try:
        data = response.json()
    except Exception:
        logger.exception("Invalid JSON from %s", base_url)
        return None, response.status_code
    if isinstance(data, dict):
        return data, response.status_code
    return None, response.status_code
