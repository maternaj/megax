"""Tests for bookmaker authentication behavior."""

from __future__ import annotations

import pytest

from megax.bookmaker.client import BookmakerClient
from megax.bookmaker.errors import BookmakerAuthError


def test_auth_request_without_credentials_raises() -> None:
    client = BookmakerClient("https://www.tipsport.cz", username=None, password=None)
    with pytest.raises(BookmakerAuthError, match="requires credentials"):
        client.fetch("/rest/contests/v1/megatipovacka/161/clients/tips?", auth=True)


def test_ensure_logged_in_false_without_credentials() -> None:
    client = BookmakerClient("https://www.tipsport.cz")
    assert client.ensure_logged_in() is False
