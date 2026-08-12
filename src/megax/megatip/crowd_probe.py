"""Build observed crowd coverage from Megatip top-3 and selective probes."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from megax.megatip.api import MegatipApi
from megax.megatip.models import CrowdObservedSnapshot, ObservedTipCoverage, RoundMatch
from megax.megatip.parse import parse_score_label

logger = logging.getLogger(__name__)

DEFAULT_MAX_SCORE = 5
DEFAULT_PROBE_DELAY_SEC = 0.3
FLOOR_PERCENTAGE = 1


def _label(home: int, away: int) -> str:
    return f"{home}:{away}"


def scores_to_infer_from_floor(
    home: int,
    away: int,
    *,
    max_score: int = DEFAULT_MAX_SCORE,
) -> list[tuple[int, int]]:
    """When (home, away) is at the 1% floor, infer the same floor along each axis.

    Example: 0:4 at 1% -> 0:5, 0:6, ... and 1:4, 2:4, ... without extra API calls.
    """
    inferred: list[tuple[int, int]] = []
    for away_goals in range(away + 1, max_score + 1):
        inferred.append((home, away_goals))
    for home_goals in range(home + 1, max_score + 1):
        pair = (home_goals, away)
        if pair not in inferred:
            inferred.append(pair)
    return inferred


def _seed_top3(match: RoundMatch) -> ObservedTipCoverage:
    tips: dict[str, int] = {}
    is_floor: dict[str, bool] = {}
    if match.popular_tips is not None:
        for tip in match.popular_tips.top3:
            tips[tip.label] = tip.percentage
            is_floor[tip.label] = tip.is_floor
    return ObservedTipCoverage(
        match_id=match.match_id,
        round_match_id=match.round_match_id,
        tips=tips,
        is_floor=is_floor,
    )


def _apply_inferred_floor(
    coverage: ObservedTipCoverage,
    home: int,
    away: int,
    *,
    max_score: int,
) -> ObservedTipCoverage:
    inferred = dict(coverage.inferred)
    floors = dict(coverage.is_floor)
    for home_goals, away_goals in scores_to_infer_from_floor(home, away, max_score=max_score):
        label = _label(home_goals, away_goals)
        if label in coverage.known_labels() or label in inferred:
            continue
        inferred[label] = FLOOR_PERCENTAGE
        floors[label] = True
    return ObservedTipCoverage(
        match_id=coverage.match_id,
        round_match_id=coverage.round_match_id,
        tips=coverage.tips,
        probed=coverage.probed,
        inferred=inferred,
        is_floor=floors,
        failed_probes=coverage.failed_probes,
    )


def _record_probe_result(
    coverage: ObservedTipCoverage,
    home: int,
    away: int,
    *,
    percentage: int,
    is_floor: bool,
    max_score: int,
) -> ObservedTipCoverage:
    label = _label(home, away)
    probed = dict(coverage.probed)
    floors = dict(coverage.is_floor)
    probed[label] = percentage
    floors[label] = is_floor
    updated = ObservedTipCoverage(
        match_id=coverage.match_id,
        round_match_id=coverage.round_match_id,
        tips=coverage.tips,
        probed=probed,
        inferred=coverage.inferred,
        is_floor=floors,
        failed_probes=coverage.failed_probes,
    )
    if is_floor and percentage <= FLOOR_PERCENTAGE:
        updated = _apply_inferred_floor(updated, home, away, max_score=max_score)
    return updated


def probe_score_percentage(
    api: MegatipApi,
    match: RoundMatch,
    home: int,
    away: int,
    coverage: ObservedTipCoverage,
    *,
    max_score: int = DEFAULT_MAX_SCORE,
) -> ObservedTipCoverage:
    label = _label(home, away)
    if label in coverage.known_labels():
        return coverage
    probe = api.fetch_popular_tip(match.round_match_id, home, away)
    if probe is None:
        logger.warning(
            "popular-tips probe failed for match_id=%s round_match_id=%s score=%s",
            match.match_id,
            match.round_match_id,
            label,
        )
        failed = coverage.failed_probes + (label,)
        return ObservedTipCoverage(
            match_id=coverage.match_id,
            round_match_id=coverage.round_match_id,
            tips=coverage.tips,
            probed=coverage.probed,
            inferred=coverage.inferred,
            is_floor=coverage.is_floor,
            failed_probes=failed,
        )
    if probe.client_tip is not None:
        return _record_probe_result(
            coverage,
            home,
            away,
            percentage=probe.client_tip.percentage,
            is_floor=probe.client_tip.is_floor,
            max_score=max_score,
        )
    if probe.queried.home == home and probe.queried.away == away:
        return _record_probe_result(
            coverage,
            home,
            away,
            percentage=probe.queried.percentage,
            is_floor=probe.queried.is_floor,
            max_score=max_score,
        )
    return coverage


def dedupe_candidates(
    candidates: list[tuple[int, int]],
    coverage: ObservedTipCoverage,
    *,
    max_score: int = DEFAULT_MAX_SCORE,
) -> list[tuple[int, int]]:
    """Drop scores already known and collapse tail probes inferrable from anchors."""
    minimized = minimize_floor_probe_candidates(candidates, max_score=max_score)
    known = coverage.known_labels()
    return [(home, away) for home, away in minimized if _label(home, away) not in known]


def minimize_floor_probe_candidates(
    candidates: list[tuple[int, int]],
    *,
    max_score: int = DEFAULT_MAX_SCORE,
) -> list[tuple[int, int]]:
    """Keep one anchor per floor tail (e.g. probe 0:4, infer 0:5+)."""
    if not candidates:
        return []
    ordered = sorted(set(candidates), key=lambda pair: (pair[0], pair[1]))
    minimized: list[tuple[int, int]] = []
    for home, away in ordered:
        dominated = False
        for anchor_home, anchor_away in minimized:
            if home == anchor_home and away > anchor_away:
                dominated = True
                break
            if away == anchor_away and home > anchor_home:
                dominated = True
                break
        if not dominated:
            minimized.append((home, away))
    return minimized


def build_observed_coverage(
    api: MegatipApi,
    *,
    round_id: int,
    candidates_by_match: dict[int, list[tuple[int, int]]] | None = None,
    only_open: bool = True,
    max_score: int = DEFAULT_MAX_SCORE,
    probe_delay_sec: float = DEFAULT_PROBE_DELAY_SEC,
) -> CrowdObservedSnapshot | None:
    """Fetch round tips and optionally probe non-top3 candidate scores."""
    snapshot = api.fetch_round_tips(round_id)
    if snapshot is None:
        return None

    matches: list[ObservedTipCoverage] = []
    for match in snapshot.round_matches:
        if only_open and not match.is_open:
            continue
        coverage = _seed_top3(match)
        raw_candidates = (candidates_by_match or {}).get(match.match_id, [])
        candidates = dedupe_candidates(raw_candidates, coverage, max_score=max_score)
        for index, (home, away) in enumerate(candidates):
            if index > 0 and probe_delay_sec > 0:
                time.sleep(probe_delay_sec)
            coverage = probe_score_percentage(
                api,
                match,
                home,
                away,
                coverage,
                max_score=max_score,
            )
        matches.append(coverage)

    return CrowdObservedSnapshot(
        round_id=round_id,
        contest_id=snapshot.contest_id,
        fetched_at=datetime.now(timezone.utc),
        matches=tuple(matches),
    )


def candidate_labels(candidates: list[tuple[int, int]]) -> list[str]:
    return [_label(home, away) for home, away in candidates]


def labels_to_candidates(labels: list[str]) -> list[tuple[int, int]]:
    parsed: list[tuple[int, int]] = []
    for label in labels:
        score = parse_score_label(label)
        if score is not None:
            parsed.append(score)
    return parsed
