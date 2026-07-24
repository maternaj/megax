"""Tests for matrix rendering helpers."""

from __future__ import annotations

from megax.gui.matrices import format_matrix_cell, render_matrix_table


def test_format_matrix_cell_shows_fair_odds_and_percent() -> None:
    assert format_matrix_cell(0.049) == "20.4 (4.9%)"
    assert format_matrix_cell(0.10) == "10.0 (10.0%)"


def test_format_matrix_cell_handles_zero() -> None:
    assert format_matrix_cell(0.0) == "— (0.0%)"


def test_render_matrix_table_includes_fair_odds() -> None:
    html = render_matrix_table(((0.049, 0.0), (0.10, 0.01)), title="P", subtitle="test")
    assert "20.4 (4.9%)" in html
    assert "10.0 (10.0%)" in html
