"""Tests for Megatip round id helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from megax.megatip.api import MegatipApi
from megax.megatip.parse import round_number_from_entry
from megax.megatip.round import actual_round_from_round_list, detect_current_round_id

FIXTURE = Path(__file__).parent / "fixtures" / "megatip" / "clients_tips_round383.json"


def test_round_number_from_entry_parses_czech_name() -> None:
    assert round_number_from_entry({"roundName": "3. kolo"}) == 3
    assert round_number_from_entry({"roundNumber": 4}) == 4


def test_actual_round_from_round_list() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    actual = actual_round_from_round_list(data["roundList"])
    assert actual is not None
    assert actual.round_id == 383
    assert actual.round_number == 3
    assert actual.status == "IN_PROGRESS"


def test_detect_current_round_id_from_round_list() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    transport = MagicMock()
    transport.fetch.return_value = data
    api = MegatipApi(transport=transport, contest_id=161, tile_id=3575, serie_id=141)
    api.fetch_tile = MagicMock(return_value=None)
    assert detect_current_round_id(api) == 383
