"""Max-entropy score matrix fit — vendored for megax (no external repo imports).

Builds a truncated score probability grid from market total, goal difference,
and devigged 1X2 probabilities via relative-entropy projection of an
independent-Poisson base prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from megax.low_score import dixon_coles_tau

MODEL_VERSION = "megax_score_prior_maxent_v1"
DEFAULT_GRID_SIZE = 10


@dataclass(frozen=True)
class ScorePriorResult:
    model_version: str
    grid_size: int
    matrix: tuple[tuple[float, ...], ...]
    input_total_mu: float
    input_goal_diff_away_home: float
    input_p_home: float
    input_p_draw: float
    input_p_away: float
    target_home_mu: float
    target_away_mu: float
    fitted_home_mu: float
    fitted_away_mu: float
    fitted_p_home: float
    fitted_p_draw: float
    fitted_p_away: float
    tail_mass_base: float
    iterations: int
    max_abs_error: float


def devig_1x2_probs(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> tuple[float, float, float] | None:
    """Proportional devig 1X2 probabilities from decimal odds."""
    if home_odds <= 0 or draw_odds <= 0 or away_odds <= 0:
        return None
    inv_home = 1.0 / home_odds
    inv_draw = 1.0 / draw_odds
    inv_away = 1.0 / away_odds
    total = inv_home + inv_draw + inv_away
    if total <= 0 or not math.isfinite(total):
        return None
    return inv_home / total, inv_draw / total, inv_away / total


def team_means_from_total_diff(
    total_mu: float,
    goal_diff_away_home: float,
) -> tuple[float, float] | None:
    """Return (home_mu, away_mu) from total goals and away-minus-home diff."""
    if not math.isfinite(total_mu) or not math.isfinite(goal_diff_away_home):
        return None
    if total_mu <= 0:
        return None
    home_mu = (total_mu - goal_diff_away_home) / 2.0
    away_mu = (total_mu + goal_diff_away_home) / 2.0
    if home_mu <= 0 or away_mu <= 0:
        return None
    return home_mu, away_mu


def _poisson_probs(mu: float, grid_size: int) -> tuple[float, ...]:
    term = math.exp(-mu)
    out = [term]
    for i in range(1, grid_size):
        term = term * mu / i
        out.append(term)
    return tuple(out)


def _base_matrix(
    home_mu: float,
    away_mu: float,
    grid_size: int,
    *,
    low_score_rho: float = 0.0,
) -> tuple[tuple[tuple[float, ...], ...], float] | None:
    if home_mu <= 0 or away_mu <= 0 or grid_size < 2:
        return None
    home = _poisson_probs(home_mu, grid_size)
    away = _poisson_probs(away_mu, grid_size)
    raw = [
        [
            h * a * dixon_coles_tau(i, j, home_mu, away_mu, low_score_rho)
            for j, a in enumerate(away)
        ]
        for i, h in enumerate(home)
    ]
    mass = sum(sum(row) for row in raw)
    if mass <= 0 or not math.isfinite(mass):
        return None
    matrix = tuple(tuple(cell / mass for cell in row) for row in raw)
    return matrix, max(0.0, min(1.0, 1.0 - mass))


def _normalize_1x2(
    p_home: float,
    p_draw: float,
    p_away: float,
) -> tuple[float, float, float] | None:
    probs = (p_home, p_draw, p_away)
    if any((not math.isfinite(p)) or p <= 0 for p in probs):
        return None
    total = sum(probs)
    if total <= 0:
        return None
    return p_home / total, p_draw / total, p_away / total


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for k in range(col, n + 1):
            aug[col][k] /= div
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            for k in range(col, n + 1):
                aug[r][k] -= factor * aug[col][k]
    return [aug[i][n] for i in range(n)]


def _features(i: int, j: int) -> tuple[float, float, float, float]:
    return (
        float(i),
        float(j),
        1.0 if i > j else 0.0,
        1.0 if i == j else 0.0,
    )


def _project(
    base: tuple[tuple[float, ...], ...],
    theta: tuple[float, float, float, float],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, float, float, float], list[list[float]]]:
    logs: list[tuple[int, int, float]] = []
    max_log = -math.inf
    for i, row in enumerate(base):
        for j, q in enumerate(row):
            if q <= 0:
                continue
            feat = _features(i, j)
            log_w = math.log(q) + sum(theta[k] * feat[k] for k in range(4))
            logs.append((i, j, log_w))
            max_log = max(max_log, log_w)

    weights = [[0.0 for _ in base[0]] for _ in base]
    total_w = 0.0
    for i, j, log_w in logs:
        w = math.exp(log_w - max_log)
        weights[i][j] = w
        total_w += w

    probs = [[cell / total_w for cell in row] for row in weights]

    means = [0.0, 0.0, 0.0, 0.0]
    second = [[0.0 for _ in range(4)] for _ in range(4)]
    for i, row in enumerate(probs):
        for j, p in enumerate(row):
            feat = _features(i, j)
            for a in range(4):
                means[a] += p * feat[a]
                for b in range(4):
                    second[a][b] += p * feat[a] * feat[b]

    cov = [[second[a][b] - means[a] * means[b] for b in range(4)] for a in range(4)]
    return tuple(tuple(row) for row in probs), tuple(means), cov


def _max_abs_delta(current: tuple[float, ...], target: tuple[float, ...]) -> float:
    return max(abs(target[i] - current[i]) for i in range(len(target)))


def _fit_matrix(
    base: tuple[tuple[float, ...], ...],
    target: tuple[float, float, float, float],
    *,
    tolerance: float,
    max_iter: int,
) -> tuple[tuple[tuple[float, ...], ...], int, float] | None:
    theta = (0.0, 0.0, 0.0, 0.0)
    matrix, current, cov = _project(base, theta)
    err = _max_abs_delta(current, target)
    if err <= tolerance:
        return matrix, 0, err

    for iteration in range(1, max_iter + 1):
        rhs = [target[i] - current[i] for i in range(4)]
        step = _solve_linear_system(cov, rhs)
        if step is None:
            return None

        accepted = False
        scale = 1.0
        while scale >= 1.0 / 1024.0:
            cand_theta = tuple(theta[i] + scale * step[i] for i in range(4))
            cand_matrix, cand_current, cand_cov = _project(base, cand_theta)
            cand_err = _max_abs_delta(cand_current, target)
            if cand_err < err:
                theta = cand_theta
                matrix = cand_matrix
                current = cand_current
                cov = cand_cov
                err = cand_err
                accepted = True
                break
            scale *= 0.5

        if not accepted:
            return None
        if err <= tolerance:
            return matrix, iteration, err

    return matrix, max_iter, err


def matrix_moments(
    matrix: tuple[tuple[float, ...], ...],
) -> tuple[float, float, float, float, float]:
    home_mu = 0.0
    away_mu = 0.0
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            home_mu += p * i
            away_mu += p * j
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
    return home_mu, away_mu, p_home, p_draw, p_away


def fit_market_score_prior(
    *,
    expected_total: float,
    goal_diff_away_home: float,
    p_home: float,
    p_draw: float,
    p_away: float,
    grid_size: int = DEFAULT_GRID_SIZE,
    tolerance: float = 1e-9,
    max_iter: int = 80,
    low_score_rho: float = 0.0,
) -> ScorePriorResult | None:
    """Fit a max-entropy score matrix to market team means and 1X2 probs."""
    means = team_means_from_total_diff(expected_total, goal_diff_away_home)
    probs = _normalize_1x2(p_home, p_draw, p_away)
    if means is None or probs is None:
        return None
    home_mu, away_mu = means
    p_home_n, p_draw_n, p_away_n = probs

    if home_mu >= grid_size - 1e-9 or away_mu >= grid_size - 1e-9:
        return None

    base_result = _base_matrix(home_mu, away_mu, grid_size, low_score_rho=low_score_rho)
    if base_result is None:
        return None
    base, tail_mass = base_result

    target = (home_mu, away_mu, p_home_n, p_draw_n)
    fitted = _fit_matrix(base, target, tolerance=tolerance, max_iter=max_iter)
    if fitted is None:
        return None
    matrix, iterations, max_abs_error = fitted
    fitted_home_mu, fitted_away_mu, fit_home, fit_draw, fit_away = matrix_moments(matrix)

    return ScorePriorResult(
        model_version=MODEL_VERSION,
        grid_size=grid_size,
        matrix=matrix,
        input_total_mu=expected_total,
        input_goal_diff_away_home=goal_diff_away_home,
        input_p_home=p_home_n,
        input_p_draw=p_draw_n,
        input_p_away=p_away_n,
        target_home_mu=home_mu,
        target_away_mu=away_mu,
        fitted_home_mu=fitted_home_mu,
        fitted_away_mu=fitted_away_mu,
        fitted_p_home=fit_home,
        fitted_p_draw=fit_draw,
        fitted_p_away=fit_away,
        tail_mass_base=tail_mass,
        iterations=iterations,
        max_abs_error=max_abs_error,
    )
