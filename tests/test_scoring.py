"""Tests for Megatipovačka scoring (examples from official rules)."""

from megax.scoring import points


def test_exact_score():
    assert points(4, 1, 4, 1) == 10


def test_six_points_winner_and_diff():
    assert points(3, 0, 4, 1) == 6
    assert points(5, 2, 4, 1) == 6


def test_six_points_draw_not_exact():
    assert points(1, 1, 0, 0) == 6


def test_four_points_winner_only():
    assert points(5, 1, 4, 1) == 4
    assert points(2, 1, 4, 1) == 4


def test_two_points_goals_only():
    assert points(2, 3, 4, 1) == 2
    assert points(1, 4, 4, 1) == 2


def test_zero_points():
    assert points(0, 2, 4, 1) == 0
    assert points(1, 3, 4, 1) == 0
