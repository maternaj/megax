"""Expected points (EV) for exact-score tips from a probability matrix."""

from __future__ import annotations

from dataclasses import dataclass

from megax.probability import ScoreMatrixResult
from megax.scoring import points


@dataclass(frozen=True)
class TipCandidate:
    home: int
    away: int
    ev: float

    @property
    def score(self) -> str:
        return format_tip(self.home, self.away)


@dataclass(frozen=True)
class EvResult:
    best: TipCandidate
    top: tuple[TipCandidate, ...]
    grid_size: int

    @property
    def top3(self) -> tuple[TipCandidate, TipCandidate, TipCandidate]:
        if len(self.top) < 3:
            raise ValueError("EV result has fewer than 3 candidates")
        return self.top[0], self.top[1], self.top[2]


def format_tip(home: int, away: int) -> str:
    return f"{home}:{away}"


def parse_tip(score: str) -> tuple[int, int] | None:
    parts = score.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        home = int(parts[0])
        away = int(parts[1])
    except ValueError:
        return None
    if home < 0 or away < 0:
        return None
    return home, away


def expected_points(
    matrix: ScoreMatrixResult | tuple[tuple[float, ...], ...],
    tip_home: int,
    tip_away: int,
) -> float:
    """EV(T) = Σ P(i,j) × points(T, i, j) over the score grid."""
    grid = matrix.matrix if isinstance(matrix, ScoreMatrixResult) else matrix
    total = 0.0
    for actual_home, row in enumerate(grid):
        for actual_away, prob in enumerate(row):
            if prob <= 0.0:
                continue
            total += prob * points(tip_home, tip_away, actual_home, actual_away)
    return total


def iter_tip_candidates(
    matrix: ScoreMatrixResult | tuple[tuple[float, ...], ...],
    *,
    max_goals: int | None = None,
) -> tuple[TipCandidate, ...]:
    grid = matrix.matrix if isinstance(matrix, ScoreMatrixResult) else matrix
    size = len(grid)
    limit = size if max_goals is None else min(max_goals + 1, size)
    candidates: list[TipCandidate] = []
    for tip_home in range(limit):
        for tip_away in range(limit):
            candidates.append(
                TipCandidate(
                    home=tip_home,
                    away=tip_away,
                    ev=expected_points(grid, tip_home, tip_away),
                )
            )
    return tuple(candidates)


def rank_tips_by_ev(
    matrix: ScoreMatrixResult | tuple[tuple[float, ...], ...],
    *,
    top_n: int = 3,
    max_goals: int | None = None,
) -> tuple[TipCandidate, ...]:
    ranked = sorted(
        iter_tip_candidates(matrix, max_goals=max_goals),
        key=lambda candidate: (-candidate.ev, -candidate.home, -candidate.away),
    )
    if top_n <= 0:
        return ()
    return tuple(ranked[:top_n])


def compute_ev(
    matrix: ScoreMatrixResult,
    *,
    top_n: int = 3,
    max_goals: int | None = None,
) -> EvResult:
    top = rank_tips_by_ev(matrix, top_n=max(top_n, 1), max_goals=max_goals)
    if not top:
        raise ValueError("No tip candidates in probability grid")
    return EvResult(best=top[0], top=top, grid_size=matrix.grid_size)
