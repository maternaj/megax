"""Tests for Megatip joker assignment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from megax.megatip.api import MegatipApi
from megax.megatip.errors import MegatipJokerError
from megax.megatip.parse import parse_round_tips
from megax.megatip.submit import current_joker_round_match_id, set_joker

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "megatip"


@dataclass
class FakeTransport:
    joker_responses: dict[tuple[str, int], dict[str, Any]]
    calls: list[tuple[str, str, int]] = field(default_factory=list)

    def fetch(self, endpoint: str, *, retry: bool = True, auth: bool = False) -> dict[str, Any] | None:
        return None

    def put(
        self,
        endpoint: str,
        body: dict[str, Any] | None = None,
        *,
        retry: bool = True,
        auth: bool = True,
    ) -> dict[str, Any] | None:
        rm_id = int(endpoint.rsplit("/", 2)[-2])
        self.calls.append(("PUT", endpoint, rm_id))
        return self.joker_responses.get(("PUT", rm_id))

    def delete(
        self,
        endpoint: str,
        *,
        retry: bool = True,
        auth: bool = True,
    ) -> dict[str, Any] | None:
        rm_id = int(endpoint.rsplit("/", 2)[-2])
        self.calls.append(("DELETE", endpoint, rm_id))
        return self.joker_responses.get(("DELETE", rm_id))


def test_current_joker_round_match_id() -> None:
    data = json.loads((FIXTURE_DIR / "clients_tips_round383.json").read_text(encoding="utf-8"))
    snapshot = parse_round_tips(data, contest_id=161, round_id=383)
    assert current_joker_round_match_id(snapshot) == 1744


def test_set_joker_moves_from_existing_match() -> None:
    transport = FakeTransport(
        joker_responses={
            ("DELETE", 1744): json.loads((FIXTURE_DIR / "joker_remove_1744.json").read_text()),
            ("PUT", 1745): {
                "jokerUsed": True,
                "freeJoker": False,
                "showJokerModal": False,
                "jokerSnackbarText": "added",
                "refreshTipPage": True,
            },
        }
    )
    api = MegatipApi(transport=transport, contest_id=161, tile_id=3575, serie_id=141)
    result = set_joker(api, 1745, current_joker_round_match_id=1744)
    assert result.joker_used is True
    assert [call[0] for call in transport.calls] == ["DELETE", "PUT"]
    assert transport.calls[0][2] == 1744
    assert transport.calls[1][2] == 1745


def test_set_joker_blocked_raises() -> None:
    transport = FakeTransport(
        joker_responses={
            ("PUT", 1745): json.loads((FIXTURE_DIR / "joker_assign_blocked.json").read_text()),
        }
    )
    api = MegatipApi(transport=transport, contest_id=161, tile_id=3575, serie_id=141)
    with pytest.raises(MegatipJokerError):
        set_joker(api, 1745, current_joker_round_match_id=None)


def test_set_joker_noop_when_already_on_match() -> None:
    transport = FakeTransport(joker_responses={})
    api = MegatipApi(transport=transport, contest_id=161, tile_id=3575, serie_id=141)
    result = set_joker(api, 1744, current_joker_round_match_id=1744)
    assert result.joker_used is True
    assert transport.calls == []
