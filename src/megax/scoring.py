"""Megatipovačka scoring rules — 10 / 6 / 4 / 2 / 0."""

from __future__ import annotations


def points(tip_home: int, tip_away: int, actual_home: int, actual_away: int) -> int:
    """Return points for a tip given the actual full-time score."""
    if tip_home == actual_home and tip_away == actual_away:
        return 10

    tip_diff = tip_home - tip_away
    actual_diff = actual_home - actual_away
    tip_total = tip_home + tip_away
    actual_total = actual_home + actual_away

    def sign(d: int) -> int:
        return (d > 0) - (d < 0)

    same_winner = sign(tip_diff) == sign(actual_diff)
    same_diff = tip_diff == actual_diff
    same_total = tip_total == actual_total
    both_draw = tip_diff == 0 and actual_diff == 0

    if both_draw:
        return 6
    if same_winner and (same_diff or same_total):
        return 6
    if same_winner:
        return 4
    if same_total:
        return 2
    return 0
