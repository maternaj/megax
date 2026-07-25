"""Simulate-driven calibration of GPP / lineup knobs before submit."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product

from megax.config import MegaxConfig, load_config
from megax.crowd import CrowdMatrixResult, build_crowd_matrix
from megax.lineup import RoundLineup, build_round_lineup, leverage_count_for_round
from megax.probability import ScoreMatrixResult, build_score_matrix_from_match
from megax.simulate import (
    AgentSpec,
    MatchSimContext,
    SimulationConfig,
    run_simulation,
)
from megax.storage import RoundRecord, load_round_record
from megax.tipsport.offer import MegaxMatch
from megax.utility import compute_match_analysis, gpp_alpha_from_field_size


@dataclass(frozen=True)
class CalibrationKnobs:
    """Optimizer tuning parameters to search."""

    gpp_ev_ratio: float
    alpha_multiplier: float
    leverage_count: int

    @property
    def label(self) -> str:
        return (
            f"ev={self.gpp_ev_ratio:.2f} "
            f"α×={self.alpha_multiplier:.2f} "
            f"lev={self.leverage_count}"
        )


ProgressCallback = Callable[[int, int, CalibrationKnobs], None]


@dataclass(frozen=True)
class MatchPCCache:
    match_id: int
    kickoff_at: datetime
    probability: ScoreMatrixResult
    crowd: CrowdMatrixResult


@dataclass(frozen=True)
class CalibrationRow:
    knobs: CalibrationKnobs
    alpha_used: float
    p_win_a: float
    p_win_b: float
    p_win_pure_ev_joker: float
    mean_pts_a: float
    mean_pts_b: float
    beats_baseline: bool

    @property
    def p_win_best(self) -> float:
        return max(self.p_win_a, self.p_win_b)


@dataclass(frozen=True)
class CalibrationResult:
    round_key: str
    match_count: int
    universes: int
    crowd_players: int
    field_size: int
    rows: tuple[CalibrationRow, ...]
    best: CalibrationRow
    baseline_row: CalibrationRow | None
    use_chalk_mode: bool


def build_match_pc_cache(
    matches: tuple[MegaxMatch, ...],
    state,
    *,
    config: MegaxConfig | None = None,
) -> tuple[MatchPCCache, ...]:
    cfg = config or load_config()
    cache: list[MatchPCCache] = []
    for match in matches:
        prob = build_score_matrix_from_match(match)
        if prob is None:
            continue
        crowd = build_crowd_matrix(
            prob,
            state.money[str(match.match_id)],
            blend_to_p=cfg.crowd_blend_to_p,
            tail_gamma=cfg.crowd_tail_gamma,
            zero_zero_delta=cfg.crowd_zero_zero_delta,
            prelec_alpha=cfg.crowd_prelec_alpha,
            zero_zero_min=cfg.crowd_zero_zero_min,
        )
        cache.append(
            MatchPCCache(
                match_id=match.match_id,
                kickoff_at=match.kickoff_at,
                probability=prob,
                crowd=crowd,
            )
        )
    return tuple(cache)


def resolve_calibration_alpha(
    field_size: int,
    *,
    alpha_multiplier: float,
    config: MegaxConfig | None = None,
) -> float:
    cfg = config or load_config()
    if cfg.gpp_alpha is not None:
        base = cfg.gpp_alpha
    else:
        base = gpp_alpha_from_field_size(field_size)
    return max(base * alpha_multiplier, 0.0)


def contexts_from_knobs(
    pc_cache: tuple[MatchPCCache, ...],
    state,
    knobs: CalibrationKnobs,
    *,
    config: MegaxConfig | None = None,
) -> tuple[MatchSimContext, ...]:
    alpha = resolve_calibration_alpha(
        state.field_size,
        alpha_multiplier=knobs.alpha_multiplier,
        config=config,
    )
    contexts: list[MatchSimContext] = []
    for item in pc_cache:
        analysis = compute_match_analysis(
            item.probability,
            item.crowd,
            field_size=state.field_size,
            gpp_alpha=alpha,
            gpp_ev_ratio=knobs.gpp_ev_ratio,
        )
        contexts.append(
            MatchSimContext(
                match_id=item.match_id,
                kickoff_at=item.kickoff_at,
                analysis=analysis,
                probability=item.probability,
                crowd=item.crowd,
            )
        )
    return tuple(contexts)


def build_calibration_agents(
    contexts: tuple[MatchSimContext, ...],
    lineup: RoundLineup,
) -> tuple[AgentSpec, ...]:
    ev_tips = {ctx.match_id: ctx.analysis.ev.best.score for ctx in contexts}
    return (
        AgentSpec(
            name="pure_ev_joker",
            tips=ev_tips,
            joker_match_id=lineup.account_a.joker_match_id,
        ),
        AgentSpec(
            name="optimizer_a",
            tips=lineup.account_a.tips_by_match(),
            joker_match_id=lineup.account_a.joker_match_id,
        ),
        AgentSpec(
            name="optimizer_b",
            tips=lineup.account_b.tips_by_match(),
            joker_match_id=lineup.account_b.joker_match_id,
        ),
    )


def evaluate_knobs(
    pc_cache: tuple[MatchPCCache, ...],
    state,
    knobs: CalibrationKnobs,
    *,
    sim_config: SimulationConfig,
    config: MegaxConfig | None = None,
) -> CalibrationRow | None:
    try:
        contexts = contexts_from_knobs(pc_cache, state, knobs, config=config)
        lineup = build_round_lineup(
            tuple(ctx.as_lineup_context() for ctx in contexts),
            leverage_count=knobs.leverage_count,
        )
    except ValueError:
        return None
    agents = build_calibration_agents(contexts, lineup)
    result = run_simulation(contexts, agents, sim_config=sim_config)
    stats = {agent.name: agent for agent in result.agents}
    p_win_a = stats["optimizer_a"].p_win
    p_win_b = stats["optimizer_b"].p_win
    p_win_joker = stats["pure_ev_joker"].p_win
    return CalibrationRow(
        knobs=knobs,
        alpha_used=resolve_calibration_alpha(
            state.field_size,
            alpha_multiplier=knobs.alpha_multiplier,
            config=config,
        ),
        p_win_a=p_win_a,
        p_win_b=p_win_b,
        p_win_pure_ev_joker=p_win_joker,
        mean_pts_a=stats["optimizer_a"].mean_points,
        mean_pts_b=stats["optimizer_b"].mean_points,
        beats_baseline=p_win_a >= p_win_joker and p_win_b >= p_win_joker,
    )


def default_knob_grid(match_count: int, *, quick: bool = False) -> tuple[CalibrationKnobs, ...]:
    default_lev = leverage_count_for_round(match_count)
    max_lev = min(3, max(default_lev + 1, 2))

    if quick:
        ev_ratios = (0.85, 0.95, 1.0)
        alpha_mults = (0.85, 1.0, 1.15)
        leverage_counts = tuple(range(0, max_lev + 1))
    else:
        ev_ratios = (0.80, 0.85, 0.90, 0.95, 1.0)
        alpha_mults = (0.70, 0.85, 1.0, 1.15, 1.30)
        leverage_counts = tuple(range(0, max_lev + 1))

    return tuple(
        CalibrationKnobs(
            gpp_ev_ratio=ev_ratio,
            alpha_multiplier=alpha_mult,
            leverage_count=lev,
        )
        for ev_ratio, alpha_mult, lev in product(ev_ratios, alpha_mults, leverage_counts)
    )


def run_calibration(
    record: RoundRecord,
    *,
    sim_config: SimulationConfig | None = None,
    grid: tuple[CalibrationKnobs, ...] | None = None,
    config: MegaxConfig | None = None,
    progress: ProgressCallback | None = None,
) -> CalibrationResult:
    cfg = config or load_config()
    sim_cfg = sim_config or SimulationConfig(field_size=record.state.field_size)
    pc_cache = build_match_pc_cache(record.matches, record.state, config=cfg)
    if len(pc_cache) != len(record.matches):
        raise ValueError("Missing probability/crowd data for one or more matches")

    knob_grid = grid or default_knob_grid(len(pc_cache))
    rows: list[CalibrationRow] = []
    total = len(knob_grid)
    for idx, knobs in enumerate(knob_grid, start=1):
        row = evaluate_knobs(
            pc_cache,
            record.state,
            knobs,
            sim_config=sim_cfg,
            config=cfg,
        )
        if row is not None:
            rows.append(row)
        if progress is not None:
            progress(idx, total, knobs)

    if not rows:
        raise ValueError("No valid knob combinations — all grid points failed GPP filters")

    best = max(rows, key=lambda row: (row.p_win_best, row.mean_pts_a + row.mean_pts_b))
    baseline_candidates = [row for row in rows if row.knobs.leverage_count == leverage_count_for_round(len(pc_cache))]
    default_ev = 0.85
    default_alpha = 1.0
    baseline_row = next(
        (
            row
            for row in baseline_candidates
            if row.knobs.gpp_ev_ratio == default_ev and row.knobs.alpha_multiplier == default_alpha
        ),
        baseline_candidates[0] if baseline_candidates else None,
    )
    use_chalk = bool(
        best.knobs.leverage_count == 0 or best.p_win_pure_ev_joker > best.p_win_best
    )

    return CalibrationResult(
        round_key=record.round_key,
        match_count=len(pc_cache),
        universes=sim_cfg.universes,
        crowd_players=(
            sim_cfg.crowd_players
            if sim_cfg.crowd_players is not None
            else min(sim_cfg.field_size, 5_000)
        ),
        field_size=sim_cfg.field_size,
        rows=tuple(rows),
        best=best,
        baseline_row=baseline_row,
        use_chalk_mode=use_chalk,
    )


def load_and_calibrate(
    round_key: str,
    *,
    sim_config: SimulationConfig | None = None,
    grid: tuple[CalibrationKnobs, ...] | None = None,
    quick: bool = False,
    progress: ProgressCallback | None = None,
) -> CalibrationResult:
    record = load_round_record(round_key)
    if record is None:
        raise FileNotFoundError(f"Round snapshot not found: {round_key}")
    if grid is None and quick:
        grid = default_knob_grid(len(record.matches), quick=True)
    elif grid is None:
        grid = default_knob_grid(len(record.matches), quick=False)
    return run_calibration(record, sim_config=sim_config, grid=grid, progress=progress)


def format_calibration_report(result: CalibrationResult) -> str:
    lines = [
        f"Round {result.round_key}: {result.match_count} matches",
        f"Grid: {len(result.rows)} combos | {result.universes:,} universes | "
        f"{result.crowd_players:,} crowd/universe | field {result.field_size:,}",
        "",
        "Recommendation",
        "-" * 40,
        f"  Best P(win):     {result.best.p_win_best:.2%}  ({result.best.knobs.label})",
        f"  Optimizer A:     {result.best.p_win_a:.2%}  mean {result.best.mean_pts_a:.1f} pts",
        f"  Optimizer B:     {result.best.p_win_b:.2%}  mean {result.best.mean_pts_b:.2f} pts",
        f"  pure_ev_joker:   {result.best.p_win_pure_ev_joker:.2%}  (baseline at best combo)",
        f"  α used:          {result.best.alpha_used:.3f}",
    ]
    if result.baseline_row is not None:
        lines.extend(
            [
                "",
                f"  Current defaults: {result.baseline_row.p_win_best:.2%} P(win)  "
                f"({result.baseline_row.knobs.label})",
                f"  Lift vs defaults: {(result.best.p_win_best - result.baseline_row.p_win_best):+.2%}",
            ]
        )
    if result.use_chalk_mode:
        lines.extend(
            [
                "",
                "  ⚠ pure_ev_joker beats optimizer on this slate — consider chalk-heavy tips",
                "    (leverage_count=0 or higher gpp_ev_ratio) unless O/U money improves C.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "  Optimizer beats pure_ev_joker at recommended knobs.",
            ]
        )

    lines.extend(
        [
            "",
            "Top 10 by P(win)",
            f"{'P(win)A':>8} {'P(win)B':>8} {'vs joker':>9}  knobs",
            "-" * 56,
        ]
    )
    ranked = sorted(
        result.rows,
        key=lambda row: (row.p_win_best, row.mean_pts_a + row.mean_pts_b),
        reverse=True,
    )
    for row in ranked[:10]:
        vs_joker = row.p_win_best - row.p_win_pure_ev_joker
        lines.append(
            f"{row.p_win_a:>7.2%} {row.p_win_b:>7.2%} {vs_joker:>+8.2%}  {row.knobs.label}"
        )
    return "\n".join(lines)


def gui_simulation_config(field_size: int) -> SimulationConfig:
    """Default simulate settings for GUI calibration (quick, reproducible)."""
    import os

    universes = int(os.getenv("MEGAX_CALIBRATE_UNIVERSES", "1500"))
    crowd_players = int(os.getenv("MEGAX_CALIBRATE_CROWD_PLAYERS", "400"))
    seed_raw = os.getenv("MEGAX_CALIBRATE_SEED", "42")
    seed = int(seed_raw) if seed_raw.strip() else 42
    return SimulationConfig(
        universes=universes,
        field_size=field_size,
        crowd_players=crowd_players,
        seed=seed,
    )


def calibration_snapshot_from_result(result: CalibrationResult):
    from megax.gui.state import CalibrationSnapshot

    best = result.best
    return CalibrationSnapshot(
        gpp_ev_ratio=float(best.knobs.gpp_ev_ratio),
        alpha_multiplier=float(best.knobs.alpha_multiplier),
        leverage_count=int(best.knobs.leverage_count),
        alpha_used=float(best.alpha_used),
        p_win_best=float(best.p_win_best),
        p_win_a=float(best.p_win_a),
        p_win_b=float(best.p_win_b),
        p_win_pure_ev_joker=float(best.p_win_pure_ev_joker),
        use_chalk_mode=bool(result.use_chalk_mode),
        calibrated_at=datetime.now(timezone.utc).isoformat(),
        universes=int(result.universes),
        grid_size=int(len(result.rows)),
    )


def build_lineup_for_knobs(
    matches: tuple[MegaxMatch, ...],
    state,
    knobs: CalibrationKnobs,
    *,
    config: MegaxConfig | None = None,
    pc_cache: tuple[MatchPCCache, ...] | None = None,
) -> RoundLineup | None:
    cfg = config or load_config()
    cache = pc_cache or build_match_pc_cache(matches, state, config=cfg)
    if len(cache) != len(matches):
        return None
    try:
        contexts = contexts_from_knobs(cache, state, knobs, config=cfg)
    except ValueError:
        return None
    return build_round_lineup(
        tuple(ctx.as_lineup_context() for ctx in contexts),
        leverage_count=knobs.leverage_count,
    )


def knobs_from_snapshot(snapshot) -> CalibrationKnobs:
    return CalibrationKnobs(
        gpp_ev_ratio=snapshot.gpp_ev_ratio,
        alpha_multiplier=snapshot.alpha_multiplier,
        leverage_count=snapshot.leverage_count,
    )
