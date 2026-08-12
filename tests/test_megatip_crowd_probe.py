"""Tests for Megatip crowd probe orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from megax.megatip.api import MegatipApi
from megax.megatip.crowd_probe import (
    build_observed_coverage,
    minimize_floor_probe_candidates,
    probe_score_percentage,
    scores_to_infer_from_floor,
)
from megax.megatip.models import ObservedTipCoverage, RoundMatch

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "megatip"


@dataclass
class FakeTransport:
    round_data: dict[str, Any]
    probes: dict[tuple[int, int, int], dict[str, Any]]

    def fetch(self, endpoint: str, *, retry: bool = True, auth: bool = False) -> dict[str, Any] | None:
        if "clients/tips" in endpoint:
            return self.round_data
        if "popular-tips" in endpoint:
            for (rm_id, home, away), payload in self.probes.items():
                needle = f"firstParticipantTip={home}&secondParticipantTip={away}"
                if str(rm_id) in endpoint and needle in endpoint:
                    return payload
        return None

    def put(
        self,
        endpoint: str,
        body: dict[str, Any] | None = None,
        *,
        retry: bool = True,
        auth: bool = True,
    ) -> dict[str, Any] | None:
        return {"ok": True}

    def delete(
        self,
        endpoint: str,
        *,
        retry: bool = True,
        auth: bool = True,
    ) -> dict[str, Any] | None:
        return {"ok": True}


def test_probe_score_percentage_skips_cached_top3() -> None:
    match = RoundMatch(
        round_match_id=1743,
        match_id=8288405,
        match_name="Test",
        kickoff_at=None,
        status="MATCH_PREMATCH",
        popular_tips=None,
    )
    coverage = ObservedTipCoverage(
        match_id=8288405,
        round_match_id=1743,
        tips={"1:2": 32},
    )
    transport = FakeTransport(round_data={}, probes={})
    api = MegatipApi(transport=transport, contest_id=161, tile_id=3575, serie_id=141)
    updated = probe_score_percentage(api, match, 1, 2, coverage)
    assert updated.probed == {}


def test_build_observed_coverage_with_probes() -> None:
    round_data = json.loads((FIXTURE_DIR / "clients_tips_round383.json").read_text(encoding="utf-8"))
    probe_payload = json.loads((FIXTURE_DIR / "popular_tips_1743_1_0.json").read_text(encoding="utf-8"))
    transport = FakeTransport(
        round_data=round_data,
        probes={(1743, 1, 0): probe_payload},
    )
    api = MegatipApi(transport=transport, contest_id=161, tile_id=3575, serie_id=141)
    snapshot = build_observed_coverage(
        api,
        round_id=383,
        candidates_by_match={8288405: [(1, 0)]},
        only_open=True,
    )
    assert snapshot is not None
    assert len(snapshot.matches) == 5
    sparta = snapshot.by_match_id()[8288405]
    assert sparta.tips["1:2"] == 32
    assert sparta.probed["1:0"] == 1
    assert sparta.is_floor["1:0"] is True
    assert sparta.inferred["2:0"] == 1
    assert "1:1" in sparta.inferred


def test_scores_to_infer_from_floor() -> None:
    inferred = scores_to_infer_from_floor(0, 4, max_score=6)
    assert (0, 5) in inferred
    assert (0, 6) in inferred
    assert (1, 4) in inferred
    assert (0, 4) not in inferred


def test_minimize_floor_probe_candidates() -> None:
    minimized = minimize_floor_probe_candidates([(0, 4), (0, 5), (0, 6), (1, 4)], max_score=6)
    assert minimized == [(0, 4)]
    assert minimize_floor_probe_candidates([(1, 0), (2, 0), (1, 1)], max_score=6) == [(1, 0)]


def test_probe_floor_infers_tail_without_extra_calls() -> None:
    match = RoundMatch(
        round_match_id=1743,
        match_id=8288405,
        match_name="Test",
        kickoff_at=None,
        status="MATCH_PREMATCH",
        popular_tips=None,
    )
    transport = FakeTransport(
        round_data={},
        probes={
            (1743, 0, 4): {
                "firstPopularTip": {"value": "1:2", "percentage": 32},
                "secondPopularTip": {"value": "1:3", "percentage": 24},
                "thirdPopularTip": {"value": "0:2", "percentage": 16},
                "clientTip": {"value": "0:4", "percentage": 1},
            }
        },
    )
    api = MegatipApi(transport=transport, contest_id=161, tile_id=3575, serie_id=141)
    coverage = ObservedTipCoverage(match_id=8288405, round_match_id=1743)
    updated = probe_score_percentage(api, match, 0, 4, coverage, max_score=6)
    assert updated.probed["0:4"] == 1
    assert updated.inferred["0:5"] == 1
    assert updated.inferred["0:6"] == 1
    assert updated.inferred["1:4"] == 1


def test_probe_failure_recorded() -> None:
    match = RoundMatch(
        round_match_id=1743,
        match_id=8288405,
        match_name="Test",
        kickoff_at=None,
        status="MATCH_PREMATCH",
        popular_tips=None,
    )
    transport = FakeTransport(round_data={}, probes={})
    api = MegatipApi(transport=transport, contest_id=161, tile_id=3575, serie_id=141)
    coverage = ObservedTipCoverage(match_id=8288405, round_match_id=1743)
    updated = probe_score_percentage(api, match, 2, 3, coverage)
    assert updated.failed_probes == ("2:3",)
