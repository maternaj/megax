"""Vectorized Monte Carlo core for round simulation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from megax.ev import parse_tip
from megax.scoring import points

ProgressCallback = Callable[[int, int], None]

_GRID = 10
_FLAT = _GRID * _GRID


def flat_index(home: int, away: int) -> int:
    return home * _GRID + away


def build_points_lut() -> np.ndarray:
    """Lookup table: points[tip_flat, actual_flat] -> score."""
    lut = np.zeros((_FLAT, _FLAT), dtype=np.int16)
    for tip_home in range(_GRID):
        for tip_away in range(_GRID):
            tip_flat = flat_index(tip_home, tip_away)
            for actual_home in range(_GRID):
                for actual_away in range(_GRID):
                    actual_flat = flat_index(actual_home, actual_away)
                    lut[tip_flat, actual_flat] = points(
                        tip_home, tip_away, actual_home, actual_away
                    )
    return lut


def matrix_to_probs(matrix: tuple[tuple[float, ...], ...]) -> np.ndarray:
    """Normalize a score matrix to a length-100 probability vector."""
    probs = np.zeros(_FLAT, dtype=np.float64)
    for home, row in enumerate(matrix):
        for away, weight in enumerate(row):
            if weight > 0.0:
                probs[flat_index(home, away)] = weight
    total = probs.sum()
    if total <= 0.0:
        probs[flat_index(0, 0)] = 1.0
    else:
        probs /= total
    return probs


@dataclass(frozen=True)
class PreparedSimulation:
    p_probs: np.ndarray
    c_probs: np.ndarray
    agent_tip_flat: np.ndarray
    joker_mult: np.ndarray
    agent_names: tuple[str, ...]
    points_lut: np.ndarray


@dataclass(frozen=True)
class VectorizedAgentStats:
    name: str
    mean_points: float
    p_win: float
    p_top_10: float
    p_top_100: float
    p_top_1000: float


@dataclass(frozen=True)
class VectorizedSimulationResult:
    universes: int
    crowd_players: int
    field_size: int
    agents: tuple[VectorizedAgentStats, ...]


def prepare_simulation(
    *,
    p_matrices: tuple[tuple[tuple[float, ...], ...], ...],
    c_matrices: tuple[tuple[tuple[float, ...], ...], ...],
    match_ids: tuple[int, ...],
    agent_names: tuple[str, ...],
    agent_tips: tuple[dict[int, str], ...],
    agent_jokers: tuple[int | None, ...],
    points_lut: np.ndarray | None = None,
) -> PreparedSimulation:
    if not p_matrices:
        raise ValueError("Cannot prepare simulation without match contexts")
    if not agent_names:
        raise ValueError("Cannot prepare simulation without agents")
    if len(p_matrices) != len(c_matrices):
        raise ValueError("Probability and crowd matrix counts must match")

    lut = points_lut if points_lut is not None else build_points_lut()
    match_index = {match_id: idx for idx, match_id in enumerate(match_ids)}
    match_count = len(p_matrices)
    agent_count = len(agent_names)

    p_probs = np.stack([matrix_to_probs(matrix) for matrix in p_matrices])
    c_probs = np.stack([matrix_to_probs(matrix) for matrix in c_matrices])

    agent_tip_flat = np.zeros((agent_count, match_count), dtype=np.int16)
    joker_mult = np.ones((agent_count, match_count), dtype=np.int16)
    for agent_idx, tips in enumerate(agent_tips):
        for match_id, tip_text in tips.items():
            match_idx = match_index.get(match_id)
            if match_idx is None:
                continue
            parsed = parse_tip(tip_text)
            if parsed is None:
                continue
            agent_tip_flat[agent_idx, match_idx] = flat_index(parsed[0], parsed[1])
        joker_match_id = agent_jokers[agent_idx]
        if joker_match_id is not None:
            joker_idx = match_index.get(joker_match_id)
            if joker_idx is not None:
                joker_mult[agent_idx, joker_idx] = 2

    return PreparedSimulation(
        p_probs=p_probs,
        c_probs=c_probs,
        agent_tip_flat=agent_tip_flat,
        joker_mult=joker_mult,
        agent_names=agent_names,
        points_lut=lut,
    )


def _default_chunk_size(universes: int, crowd_players: int) -> int:
    """Balance vectorization gains vs memory (~8 bytes per crowd cell)."""
    if crowd_players >= 2_000:
        return min(250, universes)
    if crowd_players >= 1_000:
        return min(500, universes)
    return min(2_000, universes)


def run_simulation_vectorized(
    prepared: PreparedSimulation,
    *,
    universes: int,
    field_size: int,
    crowd_players: int,
    seed: int | None = None,
    progress: ProgressCallback | None = None,
    universe_chunk: int | None = None,
) -> VectorizedSimulationResult:
    if universes <= 0:
        raise ValueError("universes must be positive")
    if crowd_players <= 0:
        raise ValueError("crowd_players must be positive")

    rng = np.random.default_rng(seed)
    lut = prepared.points_lut
    match_count = prepared.p_probs.shape[0]
    agent_count = prepared.agent_tip_flat.shape[0]
    chunk_size = universe_chunk or _default_chunk_size(universes, crowd_players)

    point_totals = np.zeros(agent_count, dtype=np.float64)
    win_totals = np.zeros(agent_count, dtype=np.float64)
    top10_totals = np.zeros(agent_count, dtype=np.float64)
    top100_totals = np.zeros(agent_count, dtype=np.float64)
    top1000_totals = np.zeros(agent_count, dtype=np.float64)

    top10_threshold = 1.0 - (10.0 / crowd_players)
    top100_threshold = 1.0 - (100.0 / crowd_players)
    top1000_threshold = 1.0 - (1000.0 / crowd_players)

    progress_every = max(1, universes // 50)
    completed = 0

    while completed < universes:
        batch = min(chunk_size, universes - completed)

        outcomes = np.empty((batch, match_count), dtype=np.int16)
        for match_idx in range(match_count):
            outcomes[:, match_idx] = rng.choice(
                _FLAT,
                size=batch,
                p=prepared.p_probs[match_idx],
            )

        crowd_tips = np.empty((batch, crowd_players, match_count), dtype=np.int16)
        for match_idx in range(match_count):
            crowd_tips[:, :, match_idx] = rng.choice(
                _FLAT,
                size=(batch, crowd_players),
                p=prepared.c_probs[match_idx],
            )

        outcomes_bc = outcomes[:, np.newaxis, :]
        crowd_points = lut[crowd_tips, outcomes_bc]
        crowd_scores = crowd_points.sum(axis=2, dtype=np.int32)

        agent_points = lut[prepared.agent_tip_flat, outcomes_bc]
        agent_points *= prepared.joker_mult
        agent_scores = agent_points.sum(axis=2, dtype=np.int32)

        crowd_max = crowd_scores.max(axis=1)
        tied_crowd = (crowd_scores == crowd_max[:, np.newaxis]).sum(axis=1)
        win_share = np.where(
            agent_scores > crowd_max[:, np.newaxis],
            1.0,
            np.where(
                agent_scores < crowd_max[:, np.newaxis],
                0.0,
                1.0 / (tied_crowd + 1)[:, np.newaxis],
            ),
        )

        better = (crowd_scores[:, np.newaxis, :] > agent_scores[:, :, np.newaxis]).sum(axis=2)
        rank = 1.0 - (better / crowd_players)

        point_totals += agent_scores.sum(axis=0)
        win_totals += win_share.sum(axis=0)
        top10_totals += (rank >= top10_threshold).sum(axis=0)
        top100_totals += (rank >= top100_threshold).sum(axis=0)
        top1000_totals += (rank >= top1000_threshold).sum(axis=0)

        completed += batch
        if progress is not None and (
            completed % progress_every == 0 or completed == universes
        ):
            progress(completed, universes)

    stats = tuple(
        VectorizedAgentStats(
            name=prepared.agent_names[agent_idx],
            mean_points=point_totals[agent_idx] / universes,
            p_win=win_totals[agent_idx] / universes,
            p_top_10=top10_totals[agent_idx] / universes,
            p_top_100=top100_totals[agent_idx] / universes,
            p_top_1000=top1000_totals[agent_idx] / universes,
        )
        for agent_idx in range(agent_count)
    )
    return VectorizedSimulationResult(
        universes=universes,
        crowd_players=crowd_players,
        field_size=field_size,
        agents=stats,
    )
