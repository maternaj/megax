"""Tests for lineup submit validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from megax.megatip.errors import LineupSubmitError
from megax.megatip.parse import parse_round_tips
from megax.megatip.submit import lineup_from_match_tips

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "megatip"


def test_lineup_from_match_tips_raises_on_unknown_match() -> None:
    data = json.loads((FIXTURE_DIR / "clients_tips_round383.json").read_text(encoding="utf-8"))
    snapshot = parse_round_tips(data, contest_id=161, round_id=383)
    with pytest.raises(LineupSubmitError) as exc:
        lineup_from_match_tips(snapshot, {999999: (1, 0)})
    assert exc.value.missing_match_ids == [999999]
