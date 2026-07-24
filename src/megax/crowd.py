"""Crowd pick distribution C(x,y) from manual money inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

from megax.crowd_shape import (
    DEFAULT_CROWD_PRELEC_ALPHA,
    DEFAULT_CROWD_TAIL_GAMMA,
    DEFAULT_CROWD_ZERO_ZERO_DELTA,
    DEFAULT_CROWD_ZERO_ZERO_MIN,
    apply_zero_zero_floor,
    group_mass,
    group_normalized_weights,
    shape_crowd_matrix,
)
from megax.probability import ScoreMatrixResult

SOFT_BOOKS = ("fortuna", "sazkabet")
FALLBACK_BOOKS = ("tipsport", "fortuna", "sazkabet")
TOTAL_BIAS_STRENGTH = 0.35
DEFAULT_CROWD_BLEND_TO_P = 0.30


@dataclass(frozen=True)
class CrowdMatrixResult:
    matrix: tuple[tuple[float, ...], ...]
    grid_size: int
    outcome_mass: tuple[float, float, float]
    outcome_mass_raw: tuple[float, float, float] | None
    blend_to_p: float
    tail_gamma: float
    zero_zero_delta: float
    prelec_alpha: float
    zero_zero_min: float
    over_share: float | None
    source: str
    note: str


def _normalize_triple(home: float, draw: float, away: float) -> tuple[float, float, float] | None:
    if any(not math.isfinite(v) or v < 0 for v in (home, draw, away)):
        return None
    total = home + draw + away
    if total <= 0:
        return None
    return home / total, draw / total, away / total


def _avg_money_field(
    money: dict[str, dict[str, float | None]],
    books: tuple[str, ...],
    field: str,
) -> float | None:
    values = [
        money[book][field]
        for book in books
        if book in money and money[book].get(field) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def outcome_mass_from_money(
    money: dict[str, dict[str, float | None]],
) -> tuple[tuple[float, float, float] | None, str]:
    soft_triples: list[tuple[float, float, float]] = []
    for book in SOFT_BOOKS:
        if book not in money:
            continue
        if not any(money[book].get(key) is not None for key in ("home", "draw", "away")):
            continue
        triple = _normalize_triple(
            float(money[book].get("home") or 0.0),
            float(money[book].get("draw") or 0.0),
            float(money[book].get("away") or 0.0),
        )
        if triple is not None:
            soft_triples.append(triple)
    if soft_triples:
        avg = tuple(
            sum(triple[i] for triple in soft_triples) / len(soft_triples)
            for i in range(3)
        )
        normalized = _normalize_triple(*avg)
        if normalized is not None:
            return normalized, "soft_money"

    for book in FALLBACK_BOOKS:
        if book not in money:
            continue
        if not any(money[book].get(key) is not None for key in ("home", "draw", "away")):
            continue
        triple = _normalize_triple(
            float(money[book].get("home") or 0.0),
            float(money[book].get("draw") or 0.0),
            float(money[book].get("away") or 0.0),
        )
        if triple is not None:
            return triple, f"{book}_money"
    return None, "missing"


def over_share_from_money(
    money: dict[str, dict[str, float | None]],
) -> float | None:
    over = _avg_money_field(money, SOFT_BOOKS, "over")
    under = _avg_money_field(money, SOFT_BOOKS, "under")
    if over is None and under is None:
        over = _avg_money_field(money, FALLBACK_BOOKS, "over")
        under = _avg_money_field(money, FALLBACK_BOOKS, "under")
    if over is None and under is None:
        return None
    if over is not None and under is not None:
        return over / (over + under)
    return over if over is not None else None


def _conditional_matrix(
    prob: ScoreMatrixResult,
    *,
    home_win: bool,
    draw: bool,
) -> tuple[tuple[float, ...], ...]:
    """Fallback within-outcome weights from objective P when shaped weights vanish."""
    grid_size = prob.grid_size
    rows: list[list[float]] = []
    for i in range(grid_size):
        row: list[float] = []
        for j in range(grid_size):
            if draw and i == j:
                row.append(prob.matrix[i][j])
            elif home_win and i > j:
                row.append(prob.matrix[i][j])
            elif (not home_win) and (not draw) and i < j:
                row.append(prob.matrix[i][j])
            else:
                row.append(0.0)
        rows.append(row)
    total = sum(sum(r) for r in rows)
    if total <= 0:
        return tuple(tuple(0.0 for _ in range(grid_size)) for _ in range(grid_size))
    return tuple(tuple(cell / total for cell in row) for row in rows)


def _group_weights_from_shape(
    prob: ScoreMatrixResult,
    shaped: tuple[tuple[float, ...], ...],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
]:
    home = group_normalized_weights(shaped, home_win=True, draw=False)
    draw = group_normalized_weights(shaped, home_win=False, draw=True)
    away = group_normalized_weights(shaped, home_win=False, draw=False)
    if group_mass(shaped, home_win=True, draw=False) <= 1e-15:
        home = _conditional_matrix(prob, home_win=True, draw=False)
    if group_mass(shaped, home_win=False, draw=True) <= 1e-15:
        draw = _conditional_matrix(prob, home_win=False, draw=True)
    if group_mass(shaped, home_win=False, draw=False) <= 1e-15:
        away = _conditional_matrix(prob, home_win=False, draw=False)
    return home, draw, away


def blend_outcome_mass(
    money_mass: tuple[float, float, float],
    prob: ScoreMatrixResult,
    *,
    blend_to_p: float = DEFAULT_CROWD_BLEND_TO_P,
) -> tuple[float, float, float]:
    """Blend bookmaker 1X2 money share toward model P(1/X/2) for a conservative crowd estimate."""
    beta = min(max(blend_to_p, 0.0), 1.0)
    p_mass = (prob.p_home, prob.p_draw, prob.p_away)
    blended = tuple(
        (1.0 - beta) * money_mass[i] + beta * p_mass[i]
        for i in range(3)
    )
    normalized = _normalize_triple(*blended)
    if normalized is None:
        return p_mass
    return normalized


def _combine_outcomes(
    home_mass: float,
    draw_mass: float,
    away_mass: float,
    home_mat: tuple[tuple[float, ...], ...],
    draw_mat: tuple[tuple[float, ...], ...],
    away_mat: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    size = len(home_mat)
    out = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            out[i][j] = (
                home_mass * home_mat[i][j]
                + draw_mass * draw_mat[i][j]
                + away_mass * away_mat[i][j]
            )
    return tuple(tuple(row) for row in out)


def _apply_total_goals_bias(
    matrix: tuple[tuple[float, ...], ...],
    over_share: float,
) -> tuple[tuple[float, ...], ...]:
    bias = over_share - 0.5
    if abs(bias) < 1e-9:
        return matrix
    weighted = []
    for i, row in enumerate(matrix):
        wrow = []
        for j, p in enumerate(row):
            total = i + j
            weight = 1.0 + TOTAL_BIAS_STRENGTH * bias * ((total - 2.5) / 2.5)
            wrow.append(p * max(weight, 1e-6))
        weighted.append(wrow)
    total = sum(sum(r) for r in weighted)
    return tuple(tuple(cell / total for cell in row) for row in weighted)


def _normalize_matrix(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    total = sum(sum(row) for row in matrix)
    if total <= 0:
        return matrix
    return tuple(tuple(cell / total for cell in row) for row in matrix)


def build_crowd_matrix(
    prob: ScoreMatrixResult,
    money: dict[str, dict[str, float | None]],
    *,
    blend_to_p: float = DEFAULT_CROWD_BLEND_TO_P,
    tail_gamma: float = DEFAULT_CROWD_TAIL_GAMMA,
    zero_zero_delta: float = DEFAULT_CROWD_ZERO_ZERO_DELTA,
    prelec_alpha: float = DEFAULT_CROWD_PRELEC_ALPHA,
    zero_zero_min: float = DEFAULT_CROWD_ZERO_ZERO_MIN,
) -> CrowdMatrixResult:
    gamma = min(max(tail_gamma, 0.0), 1.0)
    delta = min(max(zero_zero_delta, 0.0), 1.0)
    alpha = max(prelec_alpha, 1.0)
    zero_floor = max(zero_zero_min, 0.0)
    raw_mass, source = outcome_mass_from_money(money)
    beta = min(max(blend_to_p, 0.0), 1.0)
    if raw_mass is None:
        mass = (prob.p_home, prob.p_draw, prob.p_away)
        source = "fallback_odds"
        note = "Chybí peníze % — 1X2 z kurzů; tvar skóre γ/δ/Prelec z P."
        raw_for_result = None
        beta = 1.0
    else:
        mass = blend_outcome_mass(raw_mass, prob, blend_to_p=beta)
        raw_for_result = raw_mass
        note = {
            "soft_money": "1X2 z průměru Fortuna + Sazkabet.",
            "tipsport_money": "1X2 z Tipsport peněz %.",
            "fortuna_money": "1X2 z Fortuna peněz %.",
            "sazkabet_money": "1X2 z Sazkabet peněz %.",
        }.get(source, "Peníze % na 1X2.")
        if beta > 0:
            note += f" 1X2 blend β={beta:.0%} směrem k P."

    shaped = shape_crowd_matrix(
        prob.matrix,
        tail_gamma=gamma,
        zero_zero_delta=delta,
        prelec_alpha=alpha,
    )
    home_mat, draw_mat, away_mat = _group_weights_from_shape(prob, shaped)
    matrix = _combine_outcomes(mass[0], mass[1], mass[2], home_mat, draw_mat, away_mat)
    note += f" Tvar skóre: γ={gamma:.2f}, δ₀:₀={delta:.2f}, Prelec α={alpha:.2f}."
    if zero_floor > 0:
        note += f" 0:0 min {zero_floor:.1%}."

    over_share = over_share_from_money(money)
    if over_share is not None:
        matrix = _apply_total_goals_bias(matrix, over_share)
        note += f" U/O bias: over {over_share * 100:.0f}%."

    matrix = apply_zero_zero_floor(matrix, min_share=zero_floor)
    matrix = _normalize_matrix(matrix)
    return CrowdMatrixResult(
        matrix=matrix,
        grid_size=prob.grid_size,
        outcome_mass=mass,
        outcome_mass_raw=raw_for_result,
        blend_to_p=beta,
        tail_gamma=gamma,
        zero_zero_delta=delta,
        prelec_alpha=alpha,
        zero_zero_min=zero_floor,
        over_share=over_share,
        source=source,
        note=note,
    )
