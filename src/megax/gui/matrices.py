"""Heatmap rendering for score probability matrices."""

from __future__ import annotations

from html import escape


def format_matrix_cell(value: float) -> str:
    """Format probability as fair decimal odds plus percentage, e.g. 20.4 (4.9%)."""
    pct = value * 100.0
    if value <= 0.0 or pct < 0.05:
        return f"— ({pct:.1f}%)"
    fair_odds = 1.0 / value
    return f"{fair_odds:.1f} ({pct:.1f}%)"


def render_matrix_table(
    matrix: tuple[tuple[float, ...], ...],
    *,
    title: str,
    subtitle: str | None = None,
    max_goals: int = 5,
) -> str:
    size = min(max_goals + 1, len(matrix), len(matrix[0]))
    window = [row[:size] for row in matrix[:size]]
    peak = max(max(row) for row in window) if window else 1e-9
    peak = max(peak, 1e-9)

    header = "".join(f"<th>{j}</th>" for j in range(size))
    body_rows: list[str] = []
    for i, row in enumerate(window):
        cells = []
        for j, value in enumerate(row):
            pct = value * 100.0
            intensity = min(1.0, value / peak)
            alpha = 0.12 + intensity * 0.55
            score = f"{i}:{j}"
            label = format_matrix_cell(value)
            cells.append(
                f'<td class="mx-cell" style="background:rgba(78,161,255,{alpha:.2f})" '
                f'title="{escape(score)} — {escape(label)}">'
                f"{escape(label)}</td>"
            )
        body_rows.append(f"<tr><th>{i}</th>{''.join(cells)}</tr>")

    subtitle_html = f'<div class="mx-sub">{escape(subtitle)}</div>' if subtitle else ""
    return f"""
    <div class="mx-wrap">
      <div class="mx-title">{escape(title)}</div>
      {subtitle_html}
      <table class="mx-table">
        <thead><tr><th></th>{header}</tr></thead>
        <tbody>{"".join(body_rows)}</tbody>
      </table>
      <div class="mx-legend">řádky = domácí, sloupce = hosté · 0–{max_goals} gólů · kurz (pravděpodobnost %)</div>
    </div>
    """
