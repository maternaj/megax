"""Monte Carlo round simulation — truth x crowd x strategy agents."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from megax.config import MegaxConfig, load_config
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


@dataclass(frozen=True)
class SimulationResult:
    universes: int
    crowd_players: int
    field_size: int
    agents: tuple[AgentStats, ...]


def build_match_sim_contexts(
    matches: tuple[MegaxMatch, ...],
    state: RoundGuiState,
    *,
    config: MegaxConfig | None = None,
    gpp_ev_ratio: float | None = None,
    alpha_boost: float = 0.0,
) -> tuple[MatchSimContext, ...]:
    """Build per-match P/C contexts from saved matches + GUI state."""
    cfg = config or load_config()
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
            gpp_alpha=cfg.gpp_alpha,
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


def _win_share(agent_score: int, crowd_scores: list[int]) -> float:
    if not crowd_scores:
        return 1.0 if agent_score > 0 else 0.0
    best = max(crowd_scores)
    if agent_score > best:
        return 1.0
    if agent_score < best:
        return 0.0
    tied = sum(1 for score in crowd_scores if score == best) + 1
    return 1.0 / tied


def _percentile_rank(agent_score: int, crowd_scores: list[int]) -> float:
    if not crowd_scores:
        return 1.0
    better = sum(1 for score in crowd_scores if score > agent_score)
    return 1.0 - (better / len(crowd_scores))


def build_default_agents(
    contexts: tuple[MatchSimContext, ...],
    *,
    lineup: RoundLineup | None,
    state: RoundGuiState | None = None,
) -> tuple[AgentSpec, ...]:
    agents: list[AgentSpec] = [
        AgentSpec(
            name="pure_ev",
            tips={ctx.match_id: ctx.analysis.ev.best.score for ctx in contexts},
        ),
        AgentSpec(
            name="gpp",
            tips={ctx.match_id: ctx.analysis.gpp_best.score for ctx in contexts},
        ),
    ]
    if lineup is not None:
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


def run_simulation(
    contexts: tuple[MatchSimContext, ...],
    agents: tuple[AgentSpec, ...],
    *,
    sim_config: SimulationConfig | None = None,
) -> SimulationResult:
    if not contexts:
        raise ValueError("Cannot simulate without match contexts")
    if not agents:
        raise ValueError("Cannot simulate without agents")

    cfg = sim_config or SimulationConfig()
    rng = random.Random(cfg.seed)
    match_ids = tuple(ctx.match_id for ctx in contexts)
    crowd_players = cfg.crowd_players if cfg.crowd_players is not None else min(cfg.field_size, 5_000)

    totals: dict[str, list[int]] = {agent.name: [] for agent in agents}
    wins: dict[str, float] = {agent.name: 0.0 for agent in agents}
    top10: dict[str, float] = {agent.name: 0.0 for agent in agents}
    top100: dict[str, float] = {agent.name: 0.0 for agent in agents}
    top1000: dict[str, float] = {agent.name: 0.0 for agent in agents}

    for _ in range(cfg.universes):
        outcomes = {
            ctx.match_id: sample_score(rng, ctx.probability.matrix)
            for ctx in contexts
        }
        crowd_scores: list[int] = []
        for _player in range(crowd_players):
            crowd_tips = {
                ctx.match_id: _sample_tip_from_crowd(rng, ctx.crowd.matrix)
                for ctx in contexts
            }
            crowd_scores.append(
                score_round(
                    tips=crowd_tips,
                    outcomes=outcomes,
                    match_ids=match_ids,
                )
            )

        for agent in agents:
            agent_score = score_round(
                tips=agent.tips,
                outcomes=outcomes,
                match_ids=match_ids,
                joker_match_id=agent.joker_match_id,
            )
            totals[agent.name].append(agent_score)
            wins[agent.name] += _win_share(agent_score, crowd_scores)
            rank = _percentile_rank(agent_score, crowd_scores)
            if rank >= 1.0 - (10 / max(crowd_players, 1)):
                top10[agent.name] += 1.0
            if rank >= 1.0 - (100 / max(crowd_players, 1)):
                top100[agent.name] += 1.0
            if rank >= 1.0 - (1000 / max(crowd_players, 1)):
                top1000[agent.name] += 1.0

    stats = tuple(
        AgentStats(
            name=agent.name,
            mean_points=sum(totals[agent.name]) / cfg.universes,
            p_win=wins[agent.name] / cfg.universes,
            p_top_10=top10[agent.name] / cfg.universes,
            p_top_100=top100[agent.name] / cfg.universes,
            p_top_1000=top1000[agent.name] / cfg.universes,
        )
        for agent in agents
    )
    return SimulationResult(
        universes=cfg.universes,
        crowd_players=crowd_players,
        field_size=cfg.field_size,
        agents=stats,
    )


def _sample_tip_from_crowd(rng: random.Random, matrix: tuple[tuple[float, ...], ...]) -> str:
    home, away = sample_score(rng, matrix)
    return f"{home}:{away}"


def simulate_round_record(
    record: RoundRecord,
    *,
    sim_config: SimulationConfig | None = None,
    include_saved_agents: bool = True,
) -> SimulationResult:
    cfg = sim_config or SimulationConfig(field_size=record.state.field_size)
    contexts = build_match_sim_contexts(record.matches, record.state)
    if len(contexts) != len(record.matches):
        raise ValueError("Missing probability/crowd data for one or more matches")
    lineup = build_round_lineup(tuple(ctx.as_lineup_context() for ctx in contexts))
    agents = build_default_agents(
        contexts,
        lineup=lineup,
        state=record.state if include_saved_agents else None,
    )
    return run_simulation(contexts, agents, sim_config=cfg)


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
    return "\n".join(lines)


def load_and_simulate(round_key: str, *, sim_config: SimulationConfig | None = None) -> SimulationResult:
    record = load_round_record(round_key)
    if record is None:
        raise FileNotFoundError(f"Round snapshot not found: {round_key}")
    return simulate_round_record(record, sim_config=sim_config)
