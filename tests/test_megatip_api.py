"""Tests for bookmaker endpoint builders."""

from __future__ import annotations

from megax.megatip.api import (
    clients_tips_path,
    joker_path,
    popular_tips_path,
    ranking_path,
    submit_tip_path,
    tile_path,
)


def test_clients_tips_path() -> None:
    assert clients_tips_path(161, round_id=383) == (
        "/rest/contests/v1/megatipovacka/161/clients/tips?roundId=383"
    )


def test_popular_tips_path() -> None:
    assert popular_tips_path(1743, 1, 0) == (
        "/rest/megatipovacka/v1/competitions/round-matches/1743/popular-tips"
        "?firstParticipantTip=1&secondParticipantTip=0"
    )


def test_ranking_path() -> None:
    assert ranking_path(161, serie_id=141, round_id=383) == (
        "/rest/megatipovacka/v1/competitions/161/ranking"
        "?serieId=141&roundId=383&limit=50&page=1&actual=false"
    )


def test_tile_path() -> None:
    assert tile_path(3575, actual=True) == (
        "/rest/contests/v1/tiles/MEGATIPOVACKA/3575?actual=true"
    )


def test_submit_tip_path() -> None:
    assert submit_tip_path(161, 1743) == (
        "/rest/megatipovacka/v1/competitions/161/round-matches/1743/client-tips"
    )


def test_joker_path() -> None:
    assert joker_path(161, 1744) == (
        "/rest/megatipovacka/v1/competitions/161/round-matches/1744/joker"
    )
