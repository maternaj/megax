"""Tests for matrix rendering helpers."""

from __future__ import annotations

from megax.ev import expected_points
from megax.gui.matrices import (
    build_ev_grid,
    build_utility_grid,
    format_matrix_cell,
    render_matrix_table,
)
from megax.utility import utility_score


def test_format_matrix_cell_shows_fair_odds_and_percent() -> None:
    assert format_matrix_cell(0.049) == "20.4 (4.9%)"
    assert format_matrix_cell(0.10) == "10.0 (10.0%)"


def test_format_matrix_cell_handles_zero() -> None:
    assert format_matrix_cell(0.0) == "— (0.0%)"


def test_format_matrix_cell_shows_extra_metric() -> None:
    html = format_matrix_cell(0.10, extra=4.56, extra_label="EV", extra_decimals=2)
    assert "10.0 (10.0%)" in html
    assert 'EV 4.56' in html


def test_build_ev_grid_matches_expected_points() -> None:
    matrix = ((0.5, 0.5), (0.0, 0.0))
    grid = build_ev_grid(matrix, max_goals=1)
    assert grid[0][0] == expected_points(matrix, 0, 0)
    assert grid[1][1] == expected_points(matrix, 1, 1)


def test_build_utility_grid_uses_crowd_share_and_alpha() -> None:
    prob = ((0.5, 0.5), (0.0, 0.0))
    crowd = ((0.2, 0.1), (0.05, 0.0))
    grid = build_utility_grid(prob, crowd, alpha=1.1, max_goals=1)
    ev = expected_points(prob, 0, 0)
    assert grid[0][0] == utility_score(ev, 0.2, alpha=1.1)
    assert grid[1][1] is None


def test_render_matrix_table_includes_fair_odds() -> None:
    html = render_matrix_table(((0.049, 0.0), (0.10, 0.01)), title="P", subtitle="test")
    assert "20.4 (4.9%)" in html
    assert "10.0 (10.0%)" in html


def test_render_matrix_table_includes_ev_and_utility() -> None:
    prob = ((0.5, 0.5), (0.0, 0.0))
    crowd = ((0.2, 0.1), (0.05, 0.0))
    ev_grid = build_ev_grid(prob, max_goals=1)
    u_grid = build_utility_grid(prob, crowd, alpha=1.0, max_goals=1)
    p_html = render_matrix_table(
        prob,
        title="P",
        extra_values=ev_grid,
        extra_label="EV",
    )
    c_html = render_matrix_table(
        crowd,
        title="C",
        extra_values=u_grid,
        extra_label="U",
        extra_decimals=1,
    )
    assert "EV" in p_html
    assert "U" in c_html
    assert "mx-extra" in p_html
