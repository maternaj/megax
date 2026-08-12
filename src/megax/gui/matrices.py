"""Heatmap rendering for score probability matrices."""

from __future__ import annotations

from html import escape

from megax.crowd_observed import CROWD_GRID_SIZE, P_ZERO_THRESHOLD
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
    known: tuple[tuple[bool, ...], ...] | None = None,
) -> tuple[tuple[float | None, ...], ...]:
    """GPP utility U(T)=EV(T)/C(T)^α for each tip score in the visible grid."""
    ev_grid = build_ev_grid(prob, max_goals=max_goals)
    window, size = _matrix_window(crowd, max_goals=max_goals)
    known_window = None
    if known is not None:
        known_window, _ = _matrix_window(known, max_goals=max_goals)
    rows: list[tuple[float | None, ...]] = []
    for tip_home in range(size):
        row: list[float | None] = []
        for tip_away in range(size):
            share = window[tip_home][tip_away]
            if known_window is not None:
                if not known_window[tip_home][tip_away]:
                    row.append(None)
                    continue
                if share <= 0.0:
                    ev = ev_grid[tip_home][tip_away]
                    if ev <= 0:
                        row.append(None)
                    else:
                        row.append(utility_score(ev, MIN_CROWD_SHARE, alpha=alpha))
                    continue
            elif share <= 0.0:
                row.append(None)
                continue
            row.append(
                utility_score(
                    ev_grid[tip_home][tip_away],
                    max(share, MIN_CROWD_SHARE),
                    alpha=alpha,
                )
            )
        rows.append(tuple(row))
    return tuple(rows)


def render_editable_crowd_grid(
    match_id: int,
    cells: dict[str, float],
    *,
    grid_size: int = CROWD_GRID_SIZE,
    api_top3: dict[str, int] | None = None,
    prob: tuple[tuple[float, ...], ...] | None = None,
    estimated_cells: dict[str, float] | None = None,
    show_reset: bool = False,
) -> str:
    """Editable 6×6 crowd % grid — zadané (modře) vs dopočtené (zeleně)."""
    api_top3 = api_top3 or {}
    estimated_cells = estimated_cells or {}
    match_key = str(match_id)
    grid_size = min(grid_size, CROWD_GRID_SIZE)
    header = "".join(f"<th>{j}</th>" for j in range(grid_size))
    body_rows: list[str] = []
    for home in range(grid_size):
        cells_html: list[str] = []
        for away in range(grid_size):
            cell_key = f"{home}_{away}"
            label = f"{home}:{away}"
            field = f"crowd_{match_key}_{home}_{away}"
            p_hint = ""
            if prob is not None and home < len(prob) and away < len(prob[home]):
                if prob[home][away] < P_ZERO_THRESHOLD:
                    p_hint = ' title="P&lt;1% — lze 0%"'

            if cell_key in cells:
                val = cells[cell_key]
                css = "crowd-cell-zero" if val == 0 else "crowd-cell-entered"
                cells_html.append(
                    f'<td class="{css}">'
                    f'<input type="text" name="{escape(field)}" value="{escape(f"{val:g}")}" '
                    f'style="width:2.8em;text-align:center;" inputmode="decimal"'
                    f'{p_hint}></td>'
                )
            elif cell_key in estimated_cells:
                est = estimated_cells[cell_key]
                cells_html.append(
                    f'<td class="crowd-cell-computed">'
                    f'<div class="crowd-computed-val">{est:.1f}%</div>'
                    f'<input type="text" name="{escape(field)}" value="" '
                    f'placeholder="+" style="width:2.2em;text-align:center;" inputmode="decimal" '
                    f'title="Dopočteno z P — zadejte pro přepsání"{p_hint}></td>'
                )
            else:
                api_hint = ""
                if label in api_top3:
                    api_hint = f' placeholder="{api_top3[label]}" title="API top-3"'
                cells_html.append(
                    f'<td class="crowd-cell-empty">'
                    f'<input type="text" name="{escape(field)}" value="" '
                    f'style="width:2.8em;text-align:center;" inputmode="decimal"'
                    f'{api_hint}{p_hint}></td>'
                )
        body_rows.append(f"<tr><th>{home}</th>{''.join(cells_html)}</tr>")

    known = len(cells)
    total = sum(cells.values())
    est_count = len(estimated_cells)
    est_sum = sum(estimated_cells.values())
    est_note = f" · {est_count} dopočtených ({est_sum:.1f}%)" if est_count else ""
    reset_btn = ""
    if show_reset and api_top3:
        reset_btn = (
            f'<button type="submit" class="secondary" formaction="/reset-crowd-match" '
            f'name="reset_match_id" value="{match_id}" style="font-size:.72rem;padding:.2rem .45rem;">'
            f"Reset C (top-3)</button>"
        )
    return f"""
    <div class="mx-wrap">
      <div class="mx-title-row">
        <div class="mx-title">C(x,y) — tipy davu (%)</div>
        {reset_btn}
      </div>
      <div class="mx-sub">{known} zadaných · součet {total:.1f}%{est_note} · 0–{grid_size - 1} gólů</div>
      <table class="mx-table crowd-edit">
        <thead><tr><th></th>{header}</tr></thead>
        <tbody>{"".join(body_rows)}</tbody>
      </table>
      <div class="mx-legend">
        <span class="crowd-legend-entered">■ zadané</span> (API top-3 nebo ručně) ·
        <span class="crowd-legend-computed">■ dopočtené</span> =
        (100% − zadané) × P / ΣP mimo zadané. Uloží se jen vyplněná pole.
      </div>
    </div>
    """


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
