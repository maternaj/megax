"""Monte Carlo lineup optimizer — search tips A/B maximizing P(top10/top1/win)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np

from megax.ev import parse_tip
from megax.lineup import RoundLineup, build_round_lineup
from megax.simulate import (
    MatchSimContext,
    ProgressCallback,
    SimulationConfig,
    build_match_sim_contexts,
)
from megax.simulate_engine import (
    SampledUniverses,
    flat_index,
    matrix_to_probs,
    sample_simulation_universes,
    score_agents_on_sampled,
)
from megax.storage import RoundRecord

OptimizeMetric = Literal["top10", "top1", "win"]

OptimizeProgressCallback = Callable[[str, float, float, str], None]
"""phase, done_units, total_units, detail message."""


@dataclass(frozen=True)
class DualLineup:
    tips_a: dict[int, str]
    tips_b: dict[int, str]
    joker_a: int
    joker_b: int


@dataclass(frozen=True)
class AccountOptimizeStats:
    mean_points: float
    p_win: float
    p_top_10: float
    p_top_100: float
    p_top_1000: float


@dataclass(frozen=True)
class OptimizeResult:
    metric: OptimizeMetric
    objective: float
    lineup: DualLineup
    account_a: AccountOptimizeStats
    account_b: AccountOptimizeStats
    universes: int
    crowd_players: int
    field_size: int
    skipped_match_ids: tuple[int, ...]
    search_passes: int
    search_evaluations: int
    note: str


@dataclass(frozen=True)
class OptimizeUniverseCache:
    """Pre-sampled universes + index maps for fast lineup re-scoring."""

    sampled: SampledUniverses
    match_ids: tuple[int, ...]
    match_index: dict[int, int]
    field_size: int
    crowd_players: int


def estimate_optimize_evaluations(
    contexts: tuple[MatchSimContext, ...],
    *,
    top_n: int = 3,
    max_passes: int = 3,
) -> int:
    """Upper-bound hill-climb steps (each is a fast re-score on fixed universes)."""
    if not contexts:
        return 0
    n = len(contexts)
    per_pass = 0
    for ctx in contexts:
        candidate_count = len(tip_candidates(ctx, top_n=top_n))
        per_pass += max(0, candidate_count - 1) * 2
    per_pass += 2 * max(0, n - 1)
    return 1 + max_passes * per_pass


def estimate_optimize_units(
    contexts: tuple[MatchSimContext, ...],
    *,
    universes: int,
    top_n: int = 3,
    max_passes: int = 3,
) -> tuple[int, float]:
    """Progress budget: universe sampling + hill-climb re-scores."""
    eval_budget = estimate_optimize_evaluations(
        contexts,
        top_n=top_n,
        max_passes=max_passes,
    )
    # Sampling dominates; re-scores are ~100× cheaper — weight as 1 unit each.
    units = float(universes) + float(eval_budget)
    return eval_budget, max(1.0, units)


def estimate_optimize_seconds(
    match_count: int,
    *,
    universes: int,
    max_passes: int = 3,
    avg_candidates: int = 4,
) -> tuple[int, float]:
    """Rough GUI estimate: one MC sample + fast re-scores."""
    if match_count <= 0:
        return 0, 0.0
    per_pass = match_count * (2 * max(0, avg_candidates - 1) + 2) - 2
    eval_budget = 1 + max_passes * max(0, per_pass)
    sample_sec = universes * match_count * 0.00015 + 0.2
    rescore_sec = eval_budget * 0.02
    return eval_budget, max(1.0, sample_sec + rescore_sec)


def gui_optimize_config(
    field_size: int,
    *,
    universes: int | None = None,
    crowd_players: int | None = None,
    search_universes: int | None = None,
    seed: int | None = None,
) -> SimulationConfig:
    """MC settings for optimizer (single shared universe sample)."""
    import os

    from megax.simulate import gui_simulation_config

    cfg = gui_simulation_config(
        field_size,
        universes=universes,
        crowd_players=crowd_players,
        seed=seed,
    )
    if search_universes is not None:
        return SimulationConfig(
            universes=min(search_universes, cfg.universes),
            field_size=cfg.field_size,
            crowd_players=cfg.crowd_players,
            seed=cfg.seed,
        )
    return cfg


def gui_optimize_config_pair(
    field_size: int,
    *,
    universes: int | None = None,
    crowd_players: int | None = None,
    search_universes: int | None = None,
    seed: int | None = None,
) -> tuple[SimulationConfig, SimulationConfig]:
    """Backward-compatible pair — both configs now share one universe draw."""
    cfg = gui_optimize_config(
        field_size,
        universes=universes,
        crowd_players=crowd_players,
        search_universes=search_universes,
        seed=seed,
    )
    return cfg, cfg


def tip_candidates(ctx: MatchSimContext, *, top_n: int = 3) -> tuple[str, ...]:
    """Top EV (+ optional EV/C) score labels per match."""
    seen: set[str] = set()
    out: list[str] = []
    for candidate in ctx.analysis.ev.top[:top_n]:
        if candidate.score not in seen:
            out.append(candidate.score)
            seen.add(candidate.score)
    for candidate in ctx.analysis.gpp_top[:2]:
        if candidate.score not in seen:
            out.append(candidate.score)
            seen.add(candidate.score)
    if not out:
        out.append(ctx.analysis.ev.best.score)
    return tuple(out)


def _stats_from_vectorized(agent) -> AccountOptimizeStats:
    return AccountOptimizeStats(
        mean_points=agent.mean_points,
        p_win=agent.p_win,
        p_top_10=agent.p_top_10,
        p_top_100=agent.p_top_100,
        p_top_1000=agent.p_top_1000,
    )


def objective_value(
    stats_a: AccountOptimizeStats,
    stats_b: AccountOptimizeStats,
    metric: OptimizeMetric,
) -> float:
    if metric == "win":
        return (stats_a.p_win + stats_b.p_win) / 2.0
    if metric == "top1":
        return (stats_a.p_top_100 + stats_b.p_top_100) / 2.0
    return (stats_a.p_top_10 + stats_b.p_top_10) / 2.0


def build_optimize_cache(
    contexts: tuple[MatchSimContext, ...],
    *,
    sim_config: SimulationConfig,
    progress: ProgressCallback | None = None,
) -> OptimizeUniverseCache:
    """Sample truth + crowd once; hill-climb reuses these universes."""
    if not contexts:
        raise ValueError("Cannot build optimize cache without contexts")

    match_ids = tuple(ctx.match_id for ctx in contexts)
    p_probs = np.stack([matrix_to_probs(ctx.probability.matrix) for ctx in contexts])
    c_probs = np.stack([matrix_to_probs(ctx.crowd.matrix) for ctx in contexts])
    crowd_players = (
        sim_config.crowd_players
        if sim_config.crowd_players is not None
        else min(sim_config.field_size, 5_000)
    )
    sampled = sample_simulation_universes(
        p_probs,
        c_probs,
        universes=sim_config.universes,
        crowd_players=crowd_players,
        seed=sim_config.seed,
        progress=progress,
    )
    return OptimizeUniverseCache(
        sampled=sampled,
        match_ids=match_ids,
        match_index={match_id: idx for idx, match_id in enumerate(match_ids)},
        field_size=sim_config.field_size,
        crowd_players=crowd_players,
    )


def _lineup_arrays(
    cache: OptimizeUniverseCache,
    lineup: DualLineup,
) -> tuple[np.ndarray, np.ndarray]:
    match_count = len(cache.match_ids)
    agent_tip_flat = np.zeros((2, match_count), dtype=np.int16)
    joker_mult = np.ones((2, match_count), dtype=np.int16)

    for agent_idx, (tips, joker_id) in enumerate(
        ((lineup.tips_a, lineup.joker_a), (lineup.tips_b, lineup.joker_b))
    ):
        for match_id, tip_text in tips.items():
            match_idx = cache.match_index.get(match_id)
            if match_idx is None:
                continue
            parsed = parse_tip(tip_text)
            if parsed is None:
                continue
            agent_tip_flat[agent_idx, match_idx] = flat_index(parsed[0], parsed[1])
        joker_idx = cache.match_index.get(joker_id)
        if joker_idx is not None:
            joker_mult[agent_idx, joker_idx] = 2

    return agent_tip_flat, joker_mult


def score_dual_lineup(
    cache: OptimizeUniverseCache,
    lineup: DualLineup,
) -> tuple[AccountOptimizeStats, AccountOptimizeStats]:
    """Fast re-score on fixed universes (no RNG)."""
    agent_tip_flat, joker_mult = _lineup_arrays(cache, lineup)
    stats = score_agents_on_sampled(
        cache.sampled,
        agent_tip_flat=agent_tip_flat,
        joker_mult=joker_mult,
        agent_names=("mc_opt_a", "mc_opt_b"),
    )
    return _stats_from_vectorized(stats[0]), _stats_from_vectorized(stats[1])


def _lineup_from_round(lineup: RoundLineup) -> DualLineup:
    return DualLineup(
        tips_a=lineup.account_a.tips_by_match(),
        tips_b=lineup.account_b.tips_by_match(),
        joker_a=lineup.account_a.joker_match_id,
        joker_b=lineup.account_b.joker_match_id,
    )


def _b_tip_allowed(tip: str, match_id: int, tips_a: dict[int, str], candidates: dict[int, tuple[str, ...]]) -> bool:
    if tips_a.get(match_id) != tip:
        return True
    return len(candidates[match_id]) <= 1


def optimize_dual_lineup(
    contexts: tuple[MatchSimContext, ...],
    *,
    metric: OptimizeMetric = "top10",
    top_n: int = 3,
    search_config: SimulationConfig,
    final_config: SimulationConfig | None = None,
    max_passes: int = 3,
    progress: OptimizeProgressCallback | None = None,
) -> OptimizeResult:
    if not contexts:
        raise ValueError("Cannot optimize without match contexts")

    sim_config = final_config or search_config
    match_ids = tuple(ctx.match_id for ctx in contexts)
    candidates = {ctx.match_id: tip_candidates(ctx, top_n=top_n) for ctx in contexts}
    eval_budget, progress_units = estimate_optimize_units(
        contexts,
        universes=sim_config.universes,
        top_n=top_n,
        max_passes=max_passes,
    )
    universe_weight = float(sim_config.universes)

    def report(phase: str, done: float, detail: str) -> None:
        if progress is not None:
            progress(phase, done, progress_units, detail)

    def sample_progress(done: int, total: int) -> None:
        report(
            "sample",
            done,
            f"Losuji universes {done:,}/{total:,} (sdílené pro celý search)",
        )

    report("sample", 0.0, f"Losování {sim_config.universes:,} universes…")
    cache = build_optimize_cache(contexts, sim_config=sim_config, progress=sample_progress)

    seed_lineup = build_round_lineup(tuple(ctx.as_lineup_context() for ctx in contexts))
    current = _lineup_from_round(seed_lineup)
    tips_a = dict(current.tips_a)
    tips_b = dict(current.tips_b)
    joker_a = current.joker_a
    joker_b = current.joker_b

    evaluations = 0
    progress_done = universe_weight

    def evaluate() -> tuple[float, AccountOptimizeStats, AccountOptimizeStats]:
        nonlocal evaluations, progress_done
        evaluations += 1
        progress_done += 1.0
        stats_a, stats_b = score_dual_lineup(
            cache,
            DualLineup(tips_a, tips_b, joker_a, joker_b),
        )
        score = objective_value(stats_a, stats_b, metric)
        report(
            "search",
            progress_done,
            f"Hill-climb · krok {evaluations}/{eval_budget} · {metric}={score:.2%}",
        )
        return score, stats_a, stats_b

    best_score, stats_a, stats_b = evaluate()

    for pass_idx in range(max_passes):
        improved = False

        for match_id in match_ids:
            for tip in candidates[match_id]:
                if tips_a[match_id] == tip:
                    continue
                old = tips_a[match_id]
                tips_a[match_id] = tip
                score, sa, sb = evaluate()
                if score > best_score + 1e-12:
                    best_score = score
                    stats_a, stats_b = sa, sb
                    improved = True
                else:
                    tips_a[match_id] = old

        for match_id in match_ids:
            for tip in candidates[match_id]:
                if tips_b[match_id] == tip:
                    continue
                if not _b_tip_allowed(tip, match_id, tips_a, candidates):
                    continue
                old = tips_b[match_id]
                tips_b[match_id] = tip
                score, sa, sb = evaluate()
                if score > best_score + 1e-12:
                    best_score = score
                    stats_a, stats_b = sa, sb
                    improved = True
                else:
                    tips_b[match_id] = old

        for match_id in match_ids:
            if joker_a == match_id:
                continue
            old = joker_a
            joker_a = match_id
            score, sa, sb = evaluate()
            if score > best_score + 1e-12:
                best_score = score
                stats_a, stats_b = sa, sb
                improved = True
            else:
                joker_a = old

        for match_id in match_ids:
            if joker_b == match_id:
                continue
            old = joker_b
            joker_b = match_id
            score, sa, sb = evaluate()
            if score > best_score + 1e-12:
                best_score = score
                stats_a, stats_b = sa, sb
                improved = True
            else:
                joker_b = old

        if not improved:
            break

    final_lineup = DualLineup(dict(tips_a), dict(tips_b), joker_a, joker_b)
    final_a, final_b = stats_a, stats_b
    final_score = best_score
    report("done", progress_units, f"Hotovo · {evaluations} re-scores @ {sim_config.universes:,}u")

    metric_label = {"top10": "P(top10)", "top1": "P(top1)", "win": "P(win)"}[metric]
    note = (
        f"MC optimizer · {metric_label}={final_score:.2%} · "
        f"{evaluations} re-scores · shared {sim_config.universes:,} universes"
    )

    return OptimizeResult(
        metric=metric,
        objective=final_score,
        lineup=final_lineup,
        account_a=final_a,
        account_b=final_b,
        universes=sim_config.universes,
        crowd_players=cache.crowd_players,
        field_size=sim_config.field_size,
        skipped_match_ids=(),
        search_passes=max_passes,
        search_evaluations=evaluations,
        note=note,
    )


def optimize_round_record(
    record: RoundRecord,
    *,
    metric: OptimizeMetric = "top10",
    top_n: int = 3,
    search_config: SimulationConfig | None = None,
    final_config: SimulationConfig | None = None,
    progress: OptimizeProgressCallback | None = None,
) -> OptimizeResult:
    search_cfg, final_cfg = gui_optimize_config_pair(record.state.field_size)
    if search_config is not None:
        search_cfg = search_config
    if final_config is not None:
        final_cfg = final_config
    # Prefer the larger universe count if caller passed both.
    sim_cfg = final_cfg if final_cfg.universes >= search_cfg.universes else search_cfg

    contexts = build_match_sim_contexts(record.matches, record.state)
    if not contexts:
        raise ValueError("Žádný zápas s maticí P — optimalizaci nelze spustit.")

    result = optimize_dual_lineup(
        contexts,
        metric=metric,
        top_n=top_n,
        search_config=sim_cfg,
        final_config=sim_cfg,
        progress=progress,
    )
    sim_ids = {ctx.match_id for ctx in contexts}
    skipped = tuple(m.match_id for m in record.matches if m.match_id not in sim_ids)
    if skipped:
        skip_note = f" · ignorováno {len(skipped)} zápasů bez P"
        return OptimizeResult(
            metric=result.metric,
            objective=result.objective,
            lineup=result.lineup,
            account_a=result.account_a,
            account_b=result.account_b,
            universes=result.universes,
            crowd_players=result.crowd_players,
            field_size=result.field_size,
            skipped_match_ids=skipped,
            search_passes=result.search_passes,
            search_evaluations=result.search_evaluations,
            note=result.note + skip_note,
        )
    return result


def apply_optimized_lineup(state, result: OptimizeResult) -> None:
    """Write MC optimizer result into RoundGuiState accounts A/B."""
    for match_id, tip in result.lineup.tips_a.items():
        state.accounts["A"].tips[str(match_id)] = tip
    state.accounts["A"].joker_match_id = result.lineup.joker_a
    for match_id, tip in result.lineup.tips_b.items():
        state.accounts["B"].tips[str(match_id)] = tip
    state.accounts["B"].joker_match_id = result.lineup.joker_b


def apply_optimizer_snapshot(state, snapshot) -> None:
    """Write stored OptimizerSnapshot into RoundGuiState accounts A/B."""
    for match_id, tip in snapshot.tips_a.items():
        state.accounts["A"].tips[str(match_id)] = tip
    state.accounts["A"].joker_match_id = snapshot.joker_a
    for match_id, tip in snapshot.tips_b.items():
        state.accounts["B"].tips[str(match_id)] = tip
    state.accounts["B"].joker_match_id = snapshot.joker_b


def optimizer_snapshot_from_result(result: OptimizeResult):
    from megax.gui.state import OptimizerSnapshot

    return OptimizerSnapshot(
        metric=result.metric,
        objective=result.objective,
        tips_a={str(k): v for k, v in result.lineup.tips_a.items()},
        tips_b={str(k): v for k, v in result.lineup.tips_b.items()},
        joker_a=result.lineup.joker_a,
        joker_b=result.lineup.joker_b,
        p_win_a=result.account_a.p_win,
        p_top_10_a=result.account_a.p_top_10,
        p_top_100_a=result.account_a.p_top_100,
        mean_pts_a=result.account_a.mean_points,
        p_win_b=result.account_b.p_win,
        p_top_10_b=result.account_b.p_top_10,
        p_top_100_b=result.account_b.p_top_100,
        mean_pts_b=result.account_b.mean_points,
        universes=result.universes,
        crowd_players=result.crowd_players,
        field_size=result.field_size,
        skipped_match_ids=result.skipped_match_ids,
        search_evaluations=result.search_evaluations,
        optimized_at=datetime.now(timezone.utc).isoformat(),
        note=result.note,
    )
