"""Heatmap rendering for score probability matrices."""

from __future__ import annotations

from html import escape

from megax.ev import expected_points
from megax.utility import MIN_CROWD_SHARE, utility_score


def format_matrix_cell(
    value: float,
    *,
    extra: float | None = None,
    extra_label: str | None = None,
    extra_decimals: int = 2,
) -> str:
    """Format probability as fair decimal odds plus percentage, e.g. 20.4 (4.9%)."""
    pct = value * 100.0
    if value <= 0.0 or pct < 0.05:
        base = f"— ({pct:.1f}%)"
    else:
        fair_odds = 1.0 / value
        base = f"{fair_odds:.1f} ({pct:.1f}%)"
    if extra is None or extra_label is None:
        return base
    extra_text = f"{extra:.{extra_decimals}f}"
    return f'{base}<br><span class="mx-extra">{extra_label} {extra_text}</span>'


def _matrix_window(
    matrix: tuple[tuple[float, ...], ...],
    *,
    max_goals: int,
) -> tuple[tuple[tuple[float, ...], ...], int]:
    size = min(max_goals + 1, len(matrix), len(matrix[0]))
    return tuple(row[:size] for row in matrix[:size]), size


def build_ev_grid(
    matrix: tuple[tuple[float, ...], ...],
    *,
    max_goals: int = 5,
) -> tuple[tuple[float, ...], ...]:
    """Expected points for tipping each score in the visible grid."""
    window, size = _matrix_window(matrix, max_goals=max_goals)
    return tuple(
        tuple(expected_points(window, tip_home, tip_away) for tip_away in range(size))
        for tip_home in range(size)
    )


def build_utility_grid(
    prob: tuple[tuple[float, ...], ...],
    crowd: tuple[tuple[float, ...], ...],
    *,
    alpha: float,
    max_goals: int = 5,
) -> tuple[tuple[float | None, ...], ...]:
    """GPP utility U(T)=EV(T)/C(T)^α for each tip score in the visible grid."""
    ev_grid = build_ev_grid(prob, max_goals=max_goals)
    window, size = _matrix_window(crowd, max_goals=max_goals)
    rows: list[tuple[float | None, ...]] = []
    for tip_home in range(size):
        row: list[float | None] = []
        for tip_away in range(size):
            share = window[tip_home][tip_away]
            if share <= 0.0:
                row.append(None)
            else:
                row.append(
                    utility_score(
                        ev_grid[tip_home][tip_away],
                        max(share, MIN_CROWD_SHARE),
                        alpha=alpha,
                    )
                )
        rows.append(tuple(row))
    return tuple(rows)


def render_matrix_table(
    matrix: tuple[tuple[float, ...], ...],
    *,
    title: str,
    subtitle: str | None = None,
    max_goals: int = 5,
    extra_values: tuple[tuple[float | None, ...], ...] | None = None,
    extra_label: str | None = None,
    extra_decimals: int = 2,
    legend_suffix: str = "",
) -> str:
    window, size = _matrix_window(matrix, max_goals=max_goals)
    peak = max(max(row) for row in window) if window else 1e-9
    peak = max(peak, 1e-9)

    header = "".join(f"<th>{j}</th>" for j in range(size))
    body_rows: list[str] = []
    for i, row in enumerate(window):
        cells = []
        for j, value in enumerate(row):
            pct = value * 100.0
            intensity = min(1.0, value / peak)
            alpha_bg = 0.12 + intensity * 0.55
            score = f"{i}:{j}"
            extra = None
            if extra_values is not None and i < len(extra_values) and j < len(extra_values[i]):
                extra = extra_values[i][j]
            label = format_matrix_cell(
                value,
                extra=extra,
                extra_label=extra_label,
                extra_decimals=extra_decimals,
            )
            if value <= 0.0 or pct < 0.05:
                plain = f"— ({pct:.1f}%)"
            else:
                plain = f"{1.0 / value:.1f} ({pct:.1f}%)"
            title_text = plain
            if extra is not None and extra_label:
                title_text = f"{plain} · {extra_label} {extra:.{extra_decimals}f}"
            cells.append(
                f'<td class="mx-cell" style="background:rgba(78,161,255,{alpha_bg:.2f})" '
                f'title="{escape(f"{score} — {title_text}")}">'
                f"{label}</td>"
            )
        body_rows.append(f"<tr><th>{i}</th>{''.join(cells)}</tr>")

    subtitle_html = f'<div class="mx-sub">{escape(subtitle)}</div>' if subtitle else ""
    legend = f"řádky = domácí, sloupce = hosté · 0–{max_goals} gólů · kurz (pravděpodobnost %)"
    if legend_suffix:
        legend = f"{legend} · {legend_suffix}"
    return f"""
    <div class="mx-wrap">
      <div class="mx-title">{escape(title)}</div>
      {subtitle_html}
      <table class="mx-table">
        <thead><tr><th></th>{header}</tr></thead>
        <tbody>{"".join(body_rows)}</tbody>
      </table>
      <div class="mx-legend">{escape(legend)}</div>
    </div>
    """
