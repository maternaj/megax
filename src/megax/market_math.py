"""Shared market math helpers."""

from __future__ import annotations

import math


def devig_two_way(over_odds: float, under_odds: float) -> tuple[float, float] | None:
    if over_odds <= 0 or under_odds <= 0:
        return None
    inv_over = 1.0 / over_odds
    inv_under = 1.0 / under_odds
    total = inv_over + inv_under
    if total <= 0 or not math.isfinite(total):
        return None
    return inv_over / total, inv_under / total


def poisson_pmf(k: int, mu: float) -> float:
    if k < 0:
        return 0.0
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    term = math.exp(-mu)
    for i in range(1, k + 1):
        term = term * mu / i
    return term


def p_over_poisson(line: float, mu: float) -> float | None:
    if not math.isfinite(line) or not math.isfinite(mu) or mu <= 0:
        return None
    frac = round(line - math.floor(line), 2)
    if abs(frac) < 1e-9:
        threshold = int(round(line))
    elif abs(frac - 0.5) < 1e-9:
        threshold = int(math.floor(line))
    else:
        return None
    cdf = sum(poisson_pmf(k, mu) for k in range(threshold + 1))
    return 1.0 - cdf
