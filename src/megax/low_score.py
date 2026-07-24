"""Dixon–Coles low-score adjustment for independent Poisson base matrices."""

from __future__ import annotations

DEFAULT_LOW_SCORE_RHO = -0.10


def dixon_coles_tau(home: int, away: int, home_mu: float, away_mu: float, rho: float) -> float:
    """Return the Dixon–Coles τ multiplier for one score cell."""
    if rho == 0.0:
        return 1.0
    if home == 0 and away == 0:
        return 1.0 - home_mu * away_mu * rho
    if home == 0 and away == 1:
        return 1.0 + home_mu * rho
    if home == 1 and away == 0:
        return 1.0 + away_mu * rho
    if home == 1 and away == 1:
        return 1.0 - rho
    return 1.0
