"""Bookmaker REST client (Tipsport + Chance share the same backend)."""

from __future__ import annotations

import logging
from typing import Any

import cloudscraper25

from megax.bookmaker.auth import login_request
from megax.bookmaker.errors import BookmakerAuthError
from megax.bookmaker.session import (
    HttpMethod,
    create_scraper,
    default_state_file,
    exponential_backoff,
    init_web_request,
    load_scraper_state,
    request_json,
    save_scraper_state,
)

logger = logging.getLogger(__name__)


class BookmakerClient:
    def __init__(
        self,
        base_url: str,
        *,
        state_file: str | None = None,
        username: str | None = None,
        password: str | None = None,
        max_retries: int = 2,
        require_login: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.state_file = state_file or default_state_file()
        self.username = username
        self.password = password
        self.max_retries = max_retries
        self.require_login = require_login
        self._device_id: str | None = None
        self._logged_in = False
        self._last_login_status: int | None = None

    def _credentials(self) -> tuple[str, str] | None:
        if self.username and self.password:
            return self.username, self.password
        return None

    def _restore_meta(self, state) -> None:
        self._device_id = state.device_id
        self._logged_in = state.logged_in

    def _persist(self, scraper: cloudscraper25.CloudScraper) -> None:
        save_scraper_state(
            scraper,
            self.state_file,
            device_id=self._device_id,
            logged_in=self._logged_in,
        )

    def _bootstrap(self, scraper: cloudscraper25.CloudScraper) -> bool:
        if not init_web_request(scraper, self.base_url):
            return False
        creds = self._credentials()
        if creds is None:
            return not self.require_login
        username, password = creds
        body, _status, device_id = login_request(
            scraper,
            self.base_url,
            username,
            password,
            device_id=self._device_id,
        )
        self._device_id = device_id
        self._last_login_status = _status
        if body is None:
            self._logged_in = False
            return not self.require_login
        self._logged_in = True
        return True

    def _load_or_create(self) -> cloudscraper25.CloudScraper | None:
        loaded = load_scraper_state(self.state_file)
        if loaded is not None:
            scraper, state = loaded
            self._restore_meta(state)
            return scraper
        scraper = create_scraper()
        if not self._bootstrap(scraper):
            return None
        self._persist(scraper)
        return scraper

    def ensure_logged_in(self) -> bool:
        """Return True only when credentials exist and login succeeded."""
        scraper = self._load_or_create()
        if scraper is None:
            return False
        creds = self._credentials()
        if creds is None:
            return False
        if self._logged_in:
            return True
        username, password = creds
        body, _status, device_id = login_request(
            scraper,
            self.base_url,
            username,
            password,
            device_id=self._device_id,
        )
        self._device_id = device_id
        self._last_login_status = _status
        if body is None:
            self._logged_in = False
            return False
        self._logged_in = True
        self._persist(scraper)
        return True

    def _require_auth_session(self) -> None:
        creds = self._credentials()
        if creds is None:
            raise BookmakerAuthError(
                f"Authenticated API call requires credentials for {self.base_url} "
                f"(set username/password in .env)"
            )
        if not self.ensure_logged_in():
            username, _password = creds
            if self._last_login_status == 418:
                raise BookmakerAuthError(
                    f"Chance anti-bot blocked login for {username} (HTTP 418). "
                    "Credentials alone are not enough — export a browser session to "
                    "state/chance_scraper_state.json after logging in at chance.cz in Chrome."
                )
            raise BookmakerAuthError(
                f"Login failed for {username} on {self.base_url}. "
                "Check credentials or seed state/ from a browser session."
            )

    def request(
        self,
        endpoint: str,
        *,
        method: HttpMethod = "GET",
        json_body: dict[str, Any] | None = None,
        retry: bool = True,
        auth: bool = False,
    ) -> dict[str, Any] | None:
        if auth:
            try:
                self._require_auth_session()
            except BookmakerAuthError:
                raise

        loaded = load_scraper_state(self.state_file)
        if loaded is not None:
            scraper, state = loaded
            self._restore_meta(state)
            try:
                result, status = request_json(
                    scraper,
                    self.base_url,
                    endpoint,
                    method=method,
                    json_body=json_body,
                )
                if result is not None:
                    self._persist(scraper)
                    return result
                if status in (401, 403) and self._credentials() is not None:
                    self._logged_in = False
                    if self._bootstrap(scraper):
                        self._persist(scraper)
                        result, _status = request_json(
                            scraper,
                            self.base_url,
                            endpoint,
                            method=method,
                            json_body=json_body,
                        )
                        if result is not None:
                            self._persist(scraper)
                            return result
                if status == 404 and not retry:
                    return None
            except BookmakerAuthError:
                raise
            except Exception:
                logger.warning("Saved session invalid, re-bootstrapping", exc_info=True)

        if not retry:
            scraper = self._load_or_create()
            if scraper is None:
                return None
            result, _status = request_json(
                scraper,
                self.base_url,
                endpoint,
                method=method,
                json_body=json_body,
            )
            if result is not None:
                self._persist(scraper)
            return result

        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            logger.info("Bookmaker request attempt %d/%d", attempt, self.max_retries)
            scraper = create_scraper()
            if not self._bootstrap(scraper):
                if attempt < self.max_retries:
                    exponential_backoff(attempt)
                continue
            result, _status = request_json(
                scraper,
                self.base_url,
                endpoint,
                method=method,
                json_body=json_body,
            )
            if result is not None:
                self._persist(scraper)
                return result
            if attempt < self.max_retries:
                exponential_backoff(attempt)
        return None

    def fetch(self, endpoint: str, *, retry: bool = True, auth: bool = False) -> dict[str, Any] | None:
        return self.request(endpoint, method="GET", retry=retry, auth=auth)

    def put(
        self,
        endpoint: str,
        body: dict[str, Any] | None = None,
        *,
        retry: bool = True,
        auth: bool = True,
    ) -> dict[str, Any] | None:
        return self.request(endpoint, method="PUT", json_body=body, retry=retry, auth=auth)

    def delete(
        self,
        endpoint: str,
        *,
        retry: bool = True,
        auth: bool = True,
    ) -> dict[str, Any] | None:
        return self.request(endpoint, method="DELETE", retry=retry, auth=auth)
