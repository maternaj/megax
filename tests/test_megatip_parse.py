"""Tests for Megatipovačka JSON parsing."""

from __future__ import annotations

import json
from pathlib import Path

from megax.megatip.parse import (
    megatip_round_id,
    parse_popular_tip_probe,
    parse_ranking,
    parse_round_tips,
    parse_tile,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "megatip"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_megatip_round_id_mapping() -> None:
    assert megatip_round_id(1) == 381
    assert megatip_round_id(3) == 383
    assert megatip_round_id(3, offset=400) == 403


def test_round_id_from_round_list() -> None:
    from megax.megatip.parse import round_id_from_round_list

    data = _load("clients_tips_round383.json")
    resolved = round_id_from_round_list(data["roundList"], 3)
    assert resolved == 383


def test_parse_round_tips_round383() -> None:
    data = _load("clients_tips_round383.json")
    snapshot = parse_round_tips(data, contest_id=161, round_id=383)
    assert snapshot.round_id == 383
    assert len(snapshot.round_matches) == 8
    assert len(snapshot.client_tips) == 8
    sparta = next(m for m in snapshot.round_matches if m.round_match_id == 1743)
    assert sparta.match_id == 8288405
    assert sparta.match_name.startswith("Mladá Boleslav")
    assert sparta.popular_tips is not None
    assert sparta.popular_tips.top3[0].label == "1:2"
    assert sparta.popular_tips.top3[0].percentage == 32
    joker_tip = next(t for t in snapshot.client_tips if t.round_match_id == 1744)
    assert joker_tip.joker_used is True


def test_parse_popular_tip_probe_off_top3() -> None:
    data = _load("popular_tips_1743_1_0.json")
    probe = parse_popular_tip_probe(data, round_match_id=1743, home=1, away=0)
    assert probe is not None
    assert probe.client_tip is not None
    assert probe.client_tip.label == "1:0"
    assert probe.client_tip.percentage == 1
    assert probe.client_tip.is_floor is True


def test_parse_popular_tip_probe_on_top3() -> None:
    data = _load("popular_tips_1743_1_2.json")
    probe = parse_popular_tip_probe(data, round_match_id=1743, home=1, away=2)
    assert probe is not None
    assert probe.client_tip is None
    assert probe.queried.label == "1:2"
    assert probe.queried.percentage == 32


def test_parse_ranking_round383() -> None:
    data = _load("ranking_round383.json")
    ranking = parse_ranking(data, contest_id=161, serie_id=141, round_id=383)
    assert ranking.number_of_players == 100356
    assert ranking.leader_score == ranking.entries[0].score
    assert ranking.client_entry is not None
    assert ranking.client_entry.position == 11484
    assert ranking.client_entry.score == 10


def test_parse_tile_actual() -> None:
    data = _load("tile_actual.json")
    tile = parse_tile(data, tile_id=3575)
    assert tile.contest_id == 161
    assert tile.rank == 50234
    assert tile.field_size == 130074
    assert tile.points == 54
    assert tile.current_round == 3
    assert tile.total_rounds == 18


def test_parse_joker_assign() -> None:
    from megax.megatip.parse import parse_joker_response

    data = _load("joker_assign_1744.json")
    result = parse_joker_response(data)
    assert result.joker_used is True
    assert result.free_joker is False
    assert result.refresh_tip_page is True
    assert "0/1" in (result.message or "")


def test_parse_joker_remove() -> None:
    from megax.megatip.parse import parse_joker_response

    data = _load("joker_remove_1744.json")
    result = parse_joker_response(data)
    assert result.joker_used is False
    assert result.free_joker is True
