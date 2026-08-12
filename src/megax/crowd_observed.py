"""Direct sparse C(x,y) matrix — 0..5 grid, top-3 anchors + P-ratio fill."""

from __future__ import annotations

from dataclasses import dataclass

from megax.crowd import CrowdMatrixResult
from megax.probability import ScoreMatrixResult

P_ZERO_THRESHOLD = 0.01  # P(x,y) < 1% → lze nastavit 0% davu
LONGSHOT_ODDS_THRESHOLD = 100.0
PCT_HALF_BAND = 0.5
CROWD_GRID_SIZE = 6  # 0..5 gólů — stejně jako zobrazení P(x,y)
DISPLAY_MAX_GOALS = CROWD_GRID_SIZE - 1  # 0..5 inclusive


def cell_key(home: int, away: int) -> str:
    return f"{home}_{away}"


def label_to_cell_key(label: str) -> str | None:
    cleaned = label.strip().replace(" ", "")
    if not cleaned or ":" not in cleaned:
        return None
    left, right = cleaned.split(":", 1)
    try:
        return cell_key(int(left), int(right))
    except ValueError:
        return None


def cell_key_to_label(key: str) -> str:
    home, away = key.split("_", 1)
    return f"{home}:{away}"


def top3_keys_from_labels(top3: dict[str, int] | dict[str, float]) -> frozenset[str]:
    keys: set[str] = set()
    for label in top3:
        key = label_to_cell_key(label)
        if key is not None:
            keys.add(key)
    return frozenset(keys)


def prob_window(
    prob: ScoreMatrixResult,
    *,
    grid_size: int = CROWD_GRID_SIZE,
) -> tuple[tuple[float, ...], ...]:
    size = min(grid_size, prob.grid_size)
    return tuple(
        tuple(prob.matrix[i][j] for j in range(size))
        for i in range(size)
    )


@dataclass(frozen=True)
class CrowdScoreConstraint:
    """Interpretation band for one known crowd cell (pro pozdější modelování)."""

    label: str
    home: int
    away: int
    pct: float
    lo_pct: float
    hi_pct: float
    is_longshot_floor: bool
    expected_odds: float


def expected_odds_from_prob(prob: ScoreMatrixResult, home: int, away: int) -> float:
    if home >= prob.grid_size or away >= prob.grid_size:
        return float("inf")
    p = prob.matrix[home][away]
    if p <= 1e-12:
        return float("inf")
    return 1.0 / p


def crowd_score_constraint(
    prob: ScoreMatrixResult,
    home: int,
    away: int,
    pct: float,
) -> CrowdScoreConstraint:
    label = f"{home}:{away}"
    odds = expected_odds_from_prob(prob, home, away)
    if odds > LONGSHOT_ODDS_THRESHOLD or pct <= 1.0:
        return CrowdScoreConstraint(
            label=label,
            home=home,
            away=away,
            pct=pct,
            lo_pct=0.0,
            hi_pct=PCT_HALF_BAND,
            is_longshot_floor=True,
            expected_odds=odds,
        )
    return CrowdScoreConstraint(
        label=label,
        home=home,
        away=away,
        pct=pct,
        lo_pct=max(0.0, pct - PCT_HALF_BAND),
        hi_pct=pct + PCT_HALF_BAND,
        is_longshot_floor=False,
        expected_odds=odds,
    )


def _outcome_mass_from_filled(
    matrix: tuple[tuple[float, ...], ...],
    filled: tuple[tuple[bool, ...], ...],
) -> tuple[float, float, float]:
    home = draw = away = 0.0
    for i, row in enumerate(matrix):
        for j, share in enumerate(row):
            if not filled[i][j]:
                continue
            if i > j:
                home += share
            elif i == j:
                draw += share
            else:
                away += share
    total = home + draw + away
    if total <= 0:
        return 0.0, 0.0, 0.0
    return home / total, draw / total, away / total


def merge_api_top3_into_cells(
    cells: dict[str, float],
    top3: dict[str, int],
    *,
    overwrite: bool = False,
) -> dict[str, float]:
    """Fill API top-3 into crowd cells without clobbering manual entries."""
    merged = dict(cells)
    for label, pct in top3.items():
        key = label_to_cell_key(label)
        if key is None:
            continue
        if not overwrite and key in merged:
            continue
        merged[key] = float(pct)
    return merged


def estimate_unfilled_cells(
    cells: dict[str, float],
    prob: ScoreMatrixResult,
    *,
    grid_size: int = CROWD_GRID_SIZE,
) -> dict[str, float]:
    """Split (100% − sum zadaných) across empty cells by P (P zadaných vyloučena)."""
    entered_sum = sum(cells.values())
    remaining_pct = max(0.0, 100.0 - entered_sum)
    if remaining_pct <= 1e-9:
        return {}

    p_mat = prob_window(prob, grid_size=grid_size)
    entered_keys = set(cells.keys())
    p_denom = 0.0
    for home in range(grid_size):
        for away in range(grid_size):
            key = cell_key(home, away)
            if key in entered_keys:
                continue
            p_denom += p_mat[home][away]
    if p_denom <= 1e-15:
        return {}

    estimated: dict[str, float] = {}
    for home in range(grid_size):
        for away in range(grid_size):
            key = cell_key(home, away)
            if key in entered_keys:
                continue
            pct = remaining_pct * p_mat[home][away] / p_denom
            if pct > 1e-9:
                estimated[key] = pct
    return estimated


# Backward-compatible alias
estimate_non_top3_cells = estimate_unfilled_cells


def build_crowd_matrix_from_cells(
    cells: dict[str, float],
    *,
    grid_size: int = CROWD_GRID_SIZE,
    prob: ScoreMatrixResult | None = None,
    top3_keys: frozenset[str] | None = None,
    fill_from_p: bool = True,
) -> CrowdMatrixResult:
    """Build C: explicit cells + optional P-ratio fill for non-top3 empties."""
    grid_size = min(grid_size, CROWD_GRID_SIZE)
    top3_keys = top3_keys or frozenset()

    estimated_cells: dict[str, float] = {}
    if fill_from_p and prob is not None:
        estimated_cells = estimate_unfilled_cells(cells, prob, grid_size=grid_size)

    matrix = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    explicit = [[False for _ in range(grid_size)] for _ in range(grid_size)]
    estimated = [[False for _ in range(grid_size)] for _ in range(grid_size)]
    constraints: list[CrowdScoreConstraint] = []

    def _apply(key: str, pct: float, *, is_estimated: bool) -> None:
        parts = key.split("_", 1)
        if len(parts) != 2:
            return
        try:
            home, away = int(parts[0]), int(parts[1])
        except ValueError:
            return
        if home >= grid_size or away >= grid_size:
            return
        matrix[home][away] = max(pct, 0.0) / 100.0
        if is_estimated:
            estimated[home][away] = True
        else:
            explicit[home][away] = True
            if prob is not None:
                constraints.append(crowd_score_constraint(prob, home, away, pct))

    for key, pct in cells.items():
        _apply(key, pct, is_estimated=False)
    for key, pct in estimated_cells.items():
        parts = key.split("_", 1)
        if len(parts) != 2:
            continue
        try:
            home, away = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if explicit[home][away]:
            continue
        _apply(key, pct, is_estimated=True)

    matrix_tuple = tuple(tuple(row) for row in matrix)
    explicit_tuple = tuple(tuple(row) for row in explicit)
    estimated_tuple = tuple(tuple(row) for row in estimated)
    filled = tuple(
        tuple(explicit[i][j] or estimated[i][j] for j in range(grid_size))
        for i in range(grid_size)
    )

    explicit_count = sum(sum(row) for row in explicit)
    estimated_count = sum(sum(row) for row in estimated)
    total_pct = sum(cells.values()) + sum(estimated_cells.values())

    if constraints:
        band_parts = []
        for c in constraints[:6]:
            if c.is_longshot_floor:
                band_parts.append(f"{c.label}→0–0.5%")
            else:
                band_parts.append(f"{c.label}→{c.lo_pct:g}–{c.hi_pct:g}%")
        bands = "; ".join(band_parts)
        if len(constraints) > 6:
            bands += f"; +{len(constraints) - 6}"
    else:
        bands = "—"

    fill_note = ""
    if estimated_count:
        fill_note = (
            f" Dopočet: {estimated_count} buněk z "
            f"{100.0 - sum(cells.values()):.1f}% podle P (mimo zadané)."
        )

    note = (
        f"C 0..{grid_size - 1}: {explicit_count} explicitních + {estimated_count} odhad P. "
        f"Součet {total_pct:.1f}%. Pásma: {bands}.{fill_note}"
    )
    outcome_mass = _outcome_mass_from_filled(matrix_tuple, filled)

    return CrowdMatrixResult(
        matrix=matrix_tuple,
        grid_size=grid_size,
        outcome_mass=outcome_mass,
        outcome_mass_raw=None,
        blend_to_p=0.0,
        tail_gamma=0.0,
        zero_zero_delta=0.0,
        prelec_alpha=1.0,
        zero_zero_min=0.0,
        over_share=None,
        source="sparse_cells_pfill" if estimated_count else "sparse_cells",
        note=note,
        known=explicit_tuple,
        estimated=estimated_tuple,
    )


# Backward-compatible alias used by simulate/gui imports
build_crowd_matrix_from_observed = build_crowd_matrix_from_cells


def estimated_cells_from_crowd(crowd: CrowdMatrixResult) -> dict[str, float]:
    """Map cell keys to computed % for cells marked estimated."""
    if crowd.estimated is None:
        return {}
    out: dict[str, float] = {}
    for home in range(crowd.grid_size):
        for away in range(crowd.grid_size):
            if crowd.estimated[home][away]:
                out[cell_key(home, away)] = crowd.matrix[home][away] * 100.0
    return out
