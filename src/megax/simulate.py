"""Monte Carlo round simulation — truth x crowd x strategy agents."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from collections.abc import Callable
from typing import TYPE_CHECKING

ProgressCallback = Callable[[int, int], None]

from megax.config import MegaxConfig, load_config
from megax.simulate_engine import prepare_simulation, run_simulation_vectorized
from megax.crowd import CrowdMatrixResult, build_crowd_matrix
from megax.ev import parse_tip
from megax.lineup import MatchLineupContext, RoundLineup, build_round_lineup
from megax.probability import ScoreMatrixResult, build_score_matrix_from_match
from megax.scoring import points
from megax.storage import RoundRecord, load_round_record
from megax.tipsport.offer import MegaxMatch
from megax.utility import MatchTipAnalysis, compute_match_analysis

if TYPE_CHECKING:
    from megax.gui.state import RoundGuiState


@dataclass(frozen=True)
class SimulationConfig:
    universes: int = 10_000
    field_size: int = 50_000
    crowd_players: int | None = None
    seed: int | None = None
    universe_chunk: int | None = None


@dataclass(frozen=True)
class MatchSimContext:
    match_id: int
    kickoff_at: datetime
    analysis: MatchTipAnalysis
    probability: ScoreMatrixResult
    crowd: CrowdMatrixResult

    def as_lineup_context(self) -> MatchLineupContext:
        return MatchLineupContext(
            match_id=self.match_id,
            kickoff_at=self.kickoff_at,
            analysis=self.analysis,
        )


@dataclass(frozen=True)
class AgentSpec:
    name: str
    tips: dict[int, str]
    joker_match_id: int | None = None


@dataclass(frozen=True)
class AgentStats:
    name: str
    mean_points: float
    p_win: float
    p_top_10: float
    p_top_100: float
    p_top_1000: float
    tips: dict[int, str]
    joker_match_id: int | None = None


@dataclass(frozen=True)
class SimulationResult:
    universes: int
    crowd_players: int
    field_size: int
    agents: tuple[AgentStats, ...]
    matches: tuple[MegaxMatch, ...] = ()
    optimizer_note: str | None = None


def build_match_sim_contexts(
    matches: tuple[MegaxMatch, ...],
    state: RoundGuiState,
    *,
    config: MegaxConfig | None = None,
    gpp_alpha_override: float | None = None,
    gpp_ev_ratio: float | None = None,
    alpha_boost: float = 0.0,
) -> tuple[MatchSimContext, ...]:
    """Build per-match P/C contexts from saved matches + GUI state."""
    cfg = config or load_config()
    gpp_alpha = gpp_alpha_override if gpp_alpha_override is not None else cfg.gpp_alpha
    contexts: list[MatchSimContext] = []
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
        analysis = compute_match_analysis(
            prob,
            crowd,
            field_size=state.field_size,
            gpp_alpha=gpp_alpha,
            gpp_ev_ratio=gpp_ev_ratio,
            alpha_boost=alpha_boost,
        )
        contexts.append(
            MatchSimContext(
                match_id=match.match_id,
                kickoff_at=match.kickoff_at,
                analysis=analysis,
                probability=prob,
                crowd=crowd,
            )
        )
    return tuple(contexts)


def build_lineup_contexts(
    matches: tuple[MegaxMatch, ...],
    state: RoundGuiState,
    *,
    config: MegaxConfig | None = None,
    gpp_ev_ratio: float | None = None,
    alpha_boost: float = 0.0,
) -> tuple[MatchLineupContext, ...]:
    return tuple(
        ctx.as_lineup_context()
        for ctx in build_match_sim_contexts(
            matches,
            state,
            config=config,
            gpp_ev_ratio=gpp_ev_ratio,
            alpha_boost=alpha_boost,
        )
    )


def sample_score(rng: random.Random, matrix: tuple[tuple[float, ...], ...]) -> tuple[int, int]:
    weights: list[float] = []
    coords: list[tuple[int, int]] = []
    for home, row in enumerate(matrix):
        for away, prob in enumerate(row):
            if prob <= 0.0:
                continue
            weights.append(prob)
            coords.append((home, away))
    if not weights:
        return 0, 0
    return rng.choices(coords, weights=weights, k=1)[0]


def score_round(
    *,
    tips: dict[int, str],
    outcomes: dict[int, tuple[int, int]],
    match_ids: tuple[int, ...],
    joker_match_id: int | None = None,
) -> int:
    total = 0
    for match_id in match_ids:
        tip_text = tips.get(match_id)
        if not tip_text:
            continue
        parsed = parse_tip(tip_text)
        if parsed is None:
            continue
        actual = outcomes.get(match_id)
        if actual is None:
            continue
        tip_home, tip_away = parsed
        actual_home, actual_away = actual
        scored = points(tip_home, tip_away, actual_home, actual_away)
        if joker_match_id == match_id:
            scored *= 2
        total += scored
    return total



def build_default_agents(
    contexts: tuple[MatchSimContext, ...],
    *,
    lineup: RoundLineup | None,
    state: RoundGuiState | None = None,
) -> tuple[AgentSpec, ...]:
    ev_tips = {ctx.match_id: ctx.analysis.ev.best.score for ctx in contexts}
    agents: list[AgentSpec] = [
        AgentSpec(name="pure_ev", tips=ev_tips),
        AgentSpec(
            name="gpp",
            tips={ctx.match_id: ctx.analysis.gpp_best.score for ctx in contexts},
        ),
    ]
    if lineup is not None:
        agents.append(
            AgentSpec(
                name="pure_ev_joker",
                tips=ev_tips,
                joker_match_id=lineup.account_a.joker_match_id,
            )
        )
        agents.append(
            AgentSpec(
                name="optimizer_a",
                tips=lineup.account_a.tips_by_match(),
                joker_match_id=lineup.account_a.joker_match_id,
            )
        )
        agents.append(
            AgentSpec(
                name="optimizer_b",
                tips=lineup.account_b.tips_by_match(),
                joker_match_id=lineup.account_b.joker_match_id,
            )
        )
    if state is not None:
        if state.accounts["A"].tips:
            agents.append(
                AgentSpec(
                    name="saved_a",
                    tips={int(match_id): tip for match_id, tip in state.accounts["A"].tips.items() if tip},
                    joker_match_id=state.accounts["A"].joker_match_id,
                )
            )
        if state.accounts["B"].tips:
            agents.append(
                AgentSpec(
                    name="saved_b",
                    tips={int(match_id): tip for match_id, tip in state.accounts["B"].tips.items() if tip},
                    joker_match_id=state.accounts["B"].joker_match_id,
                )
            )
    return tuple(agents)


def build_optimizer_lineup(
    matches: tuple[MegaxMatch, ...],
    state: RoundGuiState,
    *,
    contexts: tuple[MatchSimContext, ...] | None = None,
) -> tuple[RoundLineup, str]:
    """Build optimizer lineup; prefer calibrated knobs when stored on the round."""
    from megax.calibrate import build_lineup_for_knobs, knobs_from_snapshot

    if state.calibration is not None:
        lineup = build_lineup_for_knobs(
            matches,
            state,
            knobs_from_snapshot(state.calibration),
        )
        if lineup is not None:
            cal = state.calibration
            note = (
                f"calibrated ev={cal.gpp_ev_ratio:.2f} "
                f"α×{cal.alpha_multiplier:.2f} lev={cal.leverage_count}"
            )
            return lineup, note

    if contexts is None:
        contexts = build_match_sim_contexts(matches, state)
    lineup = build_round_lineup(tuple(ctx.as_lineup_context() for ctx in contexts))
    return lineup, "default knobs"


def run_simulation(
    contexts: tuple[MatchSimContext, ...],
    agents: tuple[AgentSpec, ...],
    *,
    sim_config: SimulationConfig | None = None,
    progress: ProgressCallback | None = None,
) -> SimulationResult:
    if not contexts:
        raise ValueError("Cannot simulate without match contexts")
    if not agents:
        raise ValueError("Cannot simulate without agents")

    cfg = sim_config or SimulationConfig()
    crowd_players = (
        cfg.crowd_players
        if cfg.crowd_players is not None
        else min(cfg.field_size, 5_000)
    )
    prepared = prepare_simulation(
        p_matrices=tuple(ctx.probability.matrix for ctx in contexts),
        c_matrices=tuple(ctx.crowd.matrix for ctx in contexts),
        match_ids=tuple(ctx.match_id for ctx in contexts),
        agent_names=tuple(agent.name for agent in agents),
        agent_tips=tuple(agent.tips for agent in agents),
        agent_jokers=tuple(agent.joker_match_id for agent in agents),
    )
    vectorized = run_simulation_vectorized(
        prepared,
        universes=cfg.universes,
        field_size=cfg.field_size,
        crowd_players=crowd_players,
        seed=cfg.seed,
        progress=progress,
        universe_chunk=cfg.universe_chunk,
    )
    stats = tuple(
        AgentStats(
            name=vec.name,
            mean_points=vec.mean_points,
            p_win=vec.p_win,
            p_top_10=vec.p_top_10,
            p_top_100=vec.p_top_100,
            p_top_1000=vec.p_top_1000,
            tips=dict(spec.tips),
            joker_match_id=spec.joker_match_id,
        )
        for vec, spec in zip(vectorized.agents, agents, strict=True)
    )
    return SimulationResult(
        universes=vectorized.universes,
        crowd_players=vectorized.crowd_players,
        field_size=vectorized.field_size,
        agents=stats,
    )



def simulate_round_record(
    record: RoundRecord,
    *,
    sim_config: SimulationConfig | None = None,
    include_saved_agents: bool = True,
    progress: ProgressCallback | None = None,
) -> SimulationResult:
    cfg = sim_config or SimulationConfig(field_size=record.state.field_size)
    contexts = build_match_sim_contexts(record.matches, record.state)
    if len(contexts) != len(record.matches):
        raise ValueError("Missing probability/crowd data for one or more matches")
    lineup, optimizer_note = build_optimizer_lineup(
        record.matches,
        record.state,
        contexts=contexts,
    )
    agents = build_default_agents(
        contexts,
        lineup=lineup,
        state=record.state if include_saved_agents else None,
    )
    result = run_simulation(contexts, agents, sim_config=cfg, progress=progress)
    return SimulationResult(
        universes=result.universes,
        crowd_players=result.crowd_players,
        field_size=result.field_size,
        agents=result.agents,
        matches=record.matches,
        optimizer_note=optimizer_note,
    )


def _sorted_matches(matches: tuple[MegaxMatch, ...]) -> tuple[MegaxMatch, ...]:
    return tuple(sorted(matches, key=lambda match: (match.kickoff_at, match.match_id)))


def _match_short_label(match: MegaxMatch, *, width: int = 30) -> str:
    label = match.name.replace(" - ", "–") if match.name else f"{match.home}–{match.away}"
    if len(label) > width:
        return label[: width - 1] + "…"
    return label


def _format_agent_tips_section(
    agent: AgentStats,
    matches: tuple[MegaxMatch, ...],
    *,
    optimizer_note: str | None = None,
) -> list[str]:
    lines: list[str] = [agent.name]
    meta: list[str] = []
    if agent.joker_match_id is not None:
        joker = next((m for m in matches if m.match_id == agent.joker_match_id), None)
        joker_label = _match_short_label(joker) if joker else str(agent.joker_match_id)
        meta.append(f"joker {joker_label}")
    if agent.name.startswith("optimizer") and optimizer_note:
        meta.append(optimizer_note)
    if meta:
        lines[0] = f"{agent.name} ({', '.join(meta)})"
    for match in _sorted_matches(matches):
        tip = agent.tips.get(match.match_id, "—")
        lines.append(f"  {_match_short_label(match):<{30}} {tip:>5}")
    return lines


def format_simulation_report(result: SimulationResult) -> str:
    lines = [
        f"Universes: {result.universes:,}  |  crowd players/universe: {result.crowd_players:,}  |  field: {result.field_size:,}",
        "",
        f"{'Agent':<14} {'Mean pts':>9} {'P(win)':>8} {'P top10':>8} {'P top100':>9} {'P top1k':>9}",
        "-" * 62,
    ]
    for agent in result.agents:
        lines.append(
            f"{agent.name:<14} {agent.mean_points:>9.2f} {agent.p_win:>7.2%} "
            f"{agent.p_top_10:>7.2%} {agent.p_top_100:>8.2%} {agent.p_top_1000:>8.2%}"
        )

    if result.matches:
        lines.extend(["", "Tips by agent", "=" * 62])
        for agent in result.agents:
            lines.append("")
            lines.extend(
                _format_agent_tips_section(
                    agent,
                    result.matches,
                    optimizer_note=result.optimizer_note,
                )
            )
    return "\n".join(lines)


def load_and_simulate(
    round_key: str,
    *,
    sim_config: SimulationConfig | None = None,
    progress: ProgressCallback | None = None,
) -> SimulationResult:
    record = load_round_record(round_key)
    if record is None:
        raise FileNotFoundError(f"Round snapshot not found: {round_key}")
    return simulate_round_record(record, sim_config=sim_config, progress=progress)
