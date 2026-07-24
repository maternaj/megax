"""Behavioral deformations for crowd score shape (γ tail, δ 0:0, Prelec)."""

from __future__ import annotations

import math

DEFAULT_CROWD_TAIL_GAMMA = 0.50
DEFAULT_CROWD_ZERO_ZERO_DELTA = 0.20
DEFAULT_CROWD_PRELEC_ALPHA = 1.15
DEFAULT_CROWD_ZERO_ZERO_MIN = 0.015
DEFAULT_CROWD_ZERO_ZERO_TO_11 = 0.60
DEFAULT_CROWD_ZERO_ZERO_TO_10 = 0.40
MIN_PROB = 1e-15


def _outcome_group(home: int, away: int) -> str:
    if home > away:
        return "H"
    if home == away:
        return "D"
    return "A"


def _safe_cells(group: str, size: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for home in range(min(3, size)):
        for away in range(min(3, size)):
            if _outcome_group(home, away) == group:
                cells.append((home, away))
    return cells


def apply_tail_dampening(
    matrix: tuple[tuple[float, ...], ...],
    *,
    gamma: float = DEFAULT_CROWD_TAIL_GAMMA,
) -> tuple[tuple[float, ...], ...]:
    """Damp wild scores (3+ goals) and redistribute mass evenly within the outcome group."""
    gamma = min(max(gamma, 0.0), 1.0)
    size = len(matrix)
    out = [list(row) for row in matrix]
    saved_by_group = {"H": 0.0, "D": 0.0, "A": 0.0}

    for home in range(size):
        for away in range(size):
            if home >= 3 or away >= 3:
                saved = out[home][away] * (1.0 - gamma)
                out[home][away] *= gamma
                saved_by_group[_outcome_group(home, away)] += saved

    for group in ("H", "D", "A"):
        cells = _safe_cells(group, size)
        if saved_by_group[group] <= 0 or not cells:
            continue
        bump = saved_by_group[group] / len(cells)
        for home, away in cells:
            out[home][away] += bump

    return tuple(tuple(row) for row in out)


def apply_zero_zero_aversion(
    matrix: tuple[tuple[float, ...], ...],
    *,
    delta: float = DEFAULT_CROWD_ZERO_ZERO_DELTA,
    to_11: float = DEFAULT_CROWD_ZERO_ZERO_TO_11,
    to_10: float = DEFAULT_CROWD_ZERO_ZERO_TO_10,
) -> tuple[tuple[float, ...], ...]:
    """Crowd under-tips 0:0; spill freed mass mostly to 1:1 and partly to 1:0."""
    delta = min(max(delta, 0.0), 1.0)
    if len(matrix) < 2:
        return matrix

    out = [list(row) for row in matrix]
    freed = out[0][0] * (1.0 - delta)
    out[0][0] *= delta
    out[1][1] += freed * to_11
    out[1][0] += freed * to_10
    return tuple(tuple(row) for row in out)


def prelec_weight(probability: float, alpha: float) -> float:
    """Prelec probability weighting: w(p) = exp(-(-ln p)^α)."""
    if probability <= MIN_PROB:
        return 0.0
    if alpha <= 0:
        return probability
    if probability >= 1.0:
        return 1.0
    return math.exp(-((-math.log(probability)) ** alpha))


def apply_prelec(
    matrix: tuple[tuple[float, ...], ...],
    *,
    alpha: float = DEFAULT_CROWD_PRELEC_ALPHA,
    skip_cells: frozenset[tuple[int, int]] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Apply Prelec weighting cell-wise (renormalization happens later per 1X2 group)."""
    skip = skip_cells or frozenset()
    weighted: list[list[float]] = []
    for home, row in enumerate(matrix):
        wrow: list[float] = []
        for away, probability in enumerate(row):
            if (home, away) in skip:
                wrow.append(probability)
            else:
                wrow.append(prelec_weight(probability, alpha))
        weighted.append(wrow)
    return tuple(tuple(row) for row in weighted)


def apply_zero_zero_floor(
    matrix: tuple[tuple[float, ...], ...],
    *,
    min_share: float = DEFAULT_CROWD_ZERO_ZERO_MIN,
) -> tuple[tuple[float, ...], ...]:
    """Ensure at least min_share of all tips stay on 0:0 (study: ~1.5–2%)."""
    if len(matrix) == 0 or matrix[0][0] >= min_share:
        return matrix
    size = len(matrix)
    out = [list(row) for row in matrix]
    out[0][0] = min_share
    rest_total = sum(sum(row) for row in out) - min_share
    if rest_total <= MIN_PROB:
        return matrix
    scale = (1.0 - min_share) / rest_total
    for home in range(size):
        for away in range(size):
            if home == 0 and away == 0:
                continue
            out[home][away] *= scale
    return tuple(tuple(row) for row in out)


def shape_crowd_matrix(
    matrix: tuple[tuple[float, ...], ...],
    *,
    tail_gamma: float = DEFAULT_CROWD_TAIL_GAMMA,
    zero_zero_delta: float = DEFAULT_CROWD_ZERO_ZERO_DELTA,
    prelec_alpha: float = DEFAULT_CROWD_PRELEC_ALPHA,
) -> tuple[tuple[float, ...], ...]:
    shaped = apply_tail_dampening(matrix, gamma=tail_gamma)
    shaped = apply_zero_zero_aversion(shaped, delta=zero_zero_delta)
    return apply_prelec(shaped, alpha=prelec_alpha, skip_cells=frozenset({(0, 0)}))


def _group_total(
    matrix: tuple[tuple[float, ...], ...],
    *,
    home_win: bool,
    draw: bool,
) -> float:
    total = 0.0
    size = len(matrix)
    for home in range(size):
        for away in range(size):
            if draw and home == away:
                total += matrix[home][away]
            elif home_win and home > away:
                total += matrix[home][away]
            elif (not home_win) and (not draw) and home < away:
                total += matrix[home][away]
    return total


def group_mass(
    matrix: tuple[tuple[float, ...], ...],
    *,
    home_win: bool,
    draw: bool,
) -> float:
    return _group_total(matrix, home_win=home_win, draw=draw)


def group_normalized_weights(
    shaped: tuple[tuple[float, ...], ...],
    *,
    home_win: bool,
    draw: bool,
) -> tuple[tuple[float, ...], ...]:
    """Normalize shaped weights within one 1X2 outcome group."""
    size = len(shaped)
    total = _group_total(shaped, home_win=home_win, draw=draw)
    rows: list[list[float]] = []
    for home in range(size):
        row: list[float] = []
        for away in range(size):
            in_group = (
                (draw and home == away)
                or (home_win and home > away)
                or ((not home_win) and (not draw) and home < away)
            )
            if not in_group:
                row.append(0.0)
            elif total <= MIN_PROB:
                row.append(0.0)
            else:
                row.append(shaped[home][away] / total)
        rows.append(row)
    return tuple(tuple(row) for row in rows)
