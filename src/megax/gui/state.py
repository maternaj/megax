"""Persist manual GUI inputs per round window."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from megax.config import PROJECT_ROOT

DEFAULT_FIELD_SIZE = 50_000
BOOKMAKERS = ("tipsport", "fortuna", "sazkabet")
MONEY_KEYS = ("home", "draw", "away", "under", "over")
MONEY_KEY_LABELS = {
    "home": "1",
    "draw": "X",
    "away": "2",
    "under": "u",
    "over": "o",
}


@dataclass
class MegatipMatchGui:
    round_match_id: int
    top3: dict[str, int] = field(default_factory=dict)
    client_tip: str | None = None
    status: str = ""
    result_home: int | None = None
    result_away: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_match_id": self.round_match_id,
            "top3": dict(self.top3),
            "client_tip": self.client_tip,
            "status": self.status,
            "result_home": self.result_home,
            "result_away": self.result_away,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MegatipMatchGui:
        return cls(
            round_match_id=int(raw["round_match_id"]),
            top3={str(k): int(v) for k, v in (raw.get("top3") or {}).items()},
            client_tip=raw.get("client_tip"),
            status=str(raw.get("status") or ""),
            result_home=_optional_int(raw.get("result_home")),
            result_away=_optional_int(raw.get("result_away")),
        )


@dataclass
class MegatipGuiCache:
    fetched_at: str
    round_number: int | None = None
    field_size: int | None = None
    client_rank: int | None = None
    client_points: int | None = None
    leader_score: int | None = None
    can_tip: bool = True
    missing_match_ids: tuple[int, ...] = ()
    matches: dict[str, MegatipMatchGui] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fetched_at": self.fetched_at,
            "round_number": self.round_number,
            "field_size": self.field_size,
            "client_rank": self.client_rank,
            "client_points": self.client_points,
            "leader_score": self.leader_score,
            "can_tip": self.can_tip,
            "missing_match_ids": list(self.missing_match_ids),
            "matches": {
                match_id: match.to_dict() for match_id, match in self.matches.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MegatipGuiCache:
        matches_raw = raw.get("matches") or {}
        return cls(
            fetched_at=str(raw.get("fetched_at") or ""),
            round_number=_optional_int(raw.get("round_number")),
            field_size=_optional_int(raw.get("field_size")),
            client_rank=_optional_int(raw.get("client_rank")),
            client_points=_optional_int(raw.get("client_points")),
            leader_score=_optional_int(raw.get("leader_score")),
            can_tip=bool(raw.get("can_tip", True)),
            missing_match_ids=tuple(int(x) for x in (raw.get("missing_match_ids") or [])),
            matches={
                str(match_id): MegatipMatchGui.from_dict(item)
                for match_id, item in matches_raw.items()
                if isinstance(item, dict)
            },
        )


@dataclass(frozen=True)
class SimulationAgentSnapshot:
    name: str
    mean_points: float
    p_win: float
    p_top_10: float
    p_top_100: float
    p_top_1000: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean_points": self.mean_points,
            "p_win": self.p_win,
            "p_top_10": self.p_top_10,
            "p_top_100": self.p_top_100,
            "p_top_1000": self.p_top_1000,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SimulationAgentSnapshot:
        return cls(
            name=str(raw["name"]),
            mean_points=float(raw["mean_points"]),
            p_win=float(raw["p_win"]),
            p_top_10=float(raw["p_top_10"]),
            p_top_100=float(raw["p_top_100"]),
            p_top_1000=float(raw["p_top_1000"]),
        )


@dataclass
class SimulationSnapshot:
    universes: int
    crowd_players: int
    field_size: int
    simulated_at: str
    agents: tuple[SimulationAgentSnapshot, ...] = ()
    skipped_match_ids: tuple[int, ...] = ()
    note: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "universes": self.universes,
            "crowd_players": self.crowd_players,
            "field_size": self.field_size,
            "simulated_at": self.simulated_at,
            "agents": [agent.to_dict() for agent in self.agents],
            "skipped_match_ids": list(self.skipped_match_ids),
            "note": self.note,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SimulationSnapshot:
        agents_raw = raw.get("agents") or []
        skipped_raw = raw.get("skipped_match_ids") or []
        return cls(
            universes=int(raw.get("universes") or 0),
            crowd_players=int(raw.get("crowd_players") or 0),
            field_size=int(raw.get("field_size") or DEFAULT_FIELD_SIZE),
            simulated_at=str(raw.get("simulated_at") or ""),
            agents=tuple(
                SimulationAgentSnapshot.from_dict(item)
                for item in agents_raw
                if isinstance(item, dict)
            ),
            skipped_match_ids=tuple(int(x) for x in skipped_raw),
            note=raw.get("note"),
            error=raw.get("error"),
        )


@dataclass
class OptimizerSnapshot:
    metric: str
    objective: float
    tips_a: dict[str, str]
    tips_b: dict[str, str]
    joker_a: int
    joker_b: int
    p_win_a: float
    p_top_10_a: float
    p_top_100_a: float
    mean_pts_a: float
    p_win_b: float
    p_top_10_b: float
    p_top_100_b: float
    mean_pts_b: float
    universes: int
    crowd_players: int
    field_size: int
    skipped_match_ids: tuple[int, ...] = ()
    search_evaluations: int = 0
    optimized_at: str = ""
    note: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "objective": self.objective,
            "tips_a": dict(self.tips_a),
            "tips_b": dict(self.tips_b),
            "joker_a": self.joker_a,
            "joker_b": self.joker_b,
            "p_win_a": self.p_win_a,
            "p_top_10_a": self.p_top_10_a,
            "p_top_100_a": self.p_top_100_a,
            "mean_pts_a": self.mean_pts_a,
            "p_win_b": self.p_win_b,
            "p_top_10_b": self.p_top_10_b,
            "p_top_100_b": self.p_top_100_b,
            "mean_pts_b": self.mean_pts_b,
            "universes": self.universes,
            "crowd_players": self.crowd_players,
            "field_size": self.field_size,
            "skipped_match_ids": list(self.skipped_match_ids),
            "search_evaluations": self.search_evaluations,
            "optimized_at": self.optimized_at,
            "note": self.note,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> OptimizerSnapshot:
        skipped_raw = raw.get("skipped_match_ids") or []
        return cls(
            metric=str(raw.get("metric") or "top10"),
            objective=float(raw.get("objective") or 0.0),
            tips_a={str(k): str(v) for k, v in (raw.get("tips_a") or {}).items()},
            tips_b={str(k): str(v) for k, v in (raw.get("tips_b") or {}).items()},
            joker_a=int(raw["joker_a"]),
            joker_b=int(raw["joker_b"]),
            p_win_a=float(raw.get("p_win_a") or 0.0),
            p_top_10_a=float(raw.get("p_top_10_a") or 0.0),
            p_top_100_a=float(raw.get("p_top_100_a") or 0.0),
            mean_pts_a=float(raw.get("mean_pts_a") or 0.0),
            p_win_b=float(raw.get("p_win_b") or 0.0),
            p_top_10_b=float(raw.get("p_top_10_b") or 0.0),
            p_top_100_b=float(raw.get("p_top_100_b") or 0.0),
            mean_pts_b=float(raw.get("mean_pts_b") or 0.0),
            universes=int(raw.get("universes") or 0),
            crowd_players=int(raw.get("crowd_players") or 0),
            field_size=int(raw.get("field_size") or DEFAULT_FIELD_SIZE),
            skipped_match_ids=tuple(int(x) for x in skipped_raw),
            search_evaluations=int(raw.get("search_evaluations") or 0),
            optimized_at=str(raw.get("optimized_at") or ""),
            note=raw.get("note"),
            error=raw.get("error"),
        )


@dataclass
class CalibrationSnapshot:
    """Last simulate-driven calibration stored on the round."""

    gpp_ev_ratio: float
    alpha_multiplier: float
    leverage_count: int
    alpha_used: float
    p_win_best: float
    p_win_a: float
    p_win_b: float
    p_win_pure_ev_joker: float
    use_chalk_mode: bool
    calibrated_at: str
    universes: int
    grid_size: int

    @property
    def label(self) -> str:
        return (
            f"ev={self.gpp_ev_ratio:.2f} "
            f"α×={self.alpha_multiplier:.2f} "
            f"lev={self.leverage_count}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpp_ev_ratio": self.gpp_ev_ratio,
            "alpha_multiplier": self.alpha_multiplier,
            "leverage_count": self.leverage_count,
            "alpha_used": self.alpha_used,
            "p_win_best": self.p_win_best,
            "p_win_a": self.p_win_a,
            "p_win_b": self.p_win_b,
            "p_win_pure_ev_joker": self.p_win_pure_ev_joker,
            "use_chalk_mode": self.use_chalk_mode,
            "calibrated_at": self.calibrated_at,
            "universes": self.universes,
            "grid_size": self.grid_size,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CalibrationSnapshot:
        return cls(
            gpp_ev_ratio=float(raw["gpp_ev_ratio"]),
            alpha_multiplier=float(raw["alpha_multiplier"]),
            leverage_count=int(raw["leverage_count"]),
            alpha_used=float(raw["alpha_used"]),
            p_win_best=float(raw["p_win_best"]),
            p_win_a=float(raw["p_win_a"]),
            p_win_b=float(raw["p_win_b"]),
            p_win_pure_ev_joker=float(raw["p_win_pure_ev_joker"]),
            use_chalk_mode=bool(raw.get("use_chalk_mode")),
            calibrated_at=str(raw["calibrated_at"]),
            universes=int(raw.get("universes") or 0),
            grid_size=int(raw.get("grid_size") or 0),
        )


@dataclass
class AccountState:
    rank: int | None = None
    points: int = 0
    joker_match_id: int | None = None
    tips: dict[str, str] = field(default_factory=dict)


@dataclass
class RoundGuiState:
    field_size: int = DEFAULT_FIELD_SIZE
    round_id: int | None = None
    round_number: int | None = None
    accounts: dict[str, AccountState] = field(default_factory=lambda: {
        "A": AccountState(),
        "B": AccountState(),
    })
    money: dict[str, dict[str, dict[str, float | None]]] = field(default_factory=dict)
    crowd_cells: dict[str, dict[str, float]] = field(default_factory=dict)
    calibration: CalibrationSnapshot | None = None
    last_simulation: SimulationSnapshot | None = None
    last_optimization: OptimizerSnapshot | None = None
    pending_mc_job: str | None = None
    pending_mc_started_at: str | None = None
    megatip: MegatipGuiCache | None = None

    def ensure_match(self, match_id: int) -> None:
        key = str(match_id)
        if key not in self.money:
            self.money[key] = {
                book: {k: None for k in MONEY_KEYS}
                for book in BOOKMAKERS
            }
        if key not in self.crowd_cells:
            self.crowd_cells[key] = {}

    def crowd_cells_for_match(self, match_id: int) -> dict[str, float]:
        return dict(self.crowd_cells.get(str(match_id), {}))

    def top3_cell_keys(self, match_id: int) -> frozenset[str]:
        from megax.crowd_observed import top3_keys_from_labels

        if self.megatip is None:
            return frozenset()
        match = self.megatip.matches.get(str(match_id))
        if match is None or not match.top3:
            return frozenset()
        return top3_keys_from_labels(match.top3)

    def seed_crowd_from_megatip(self, match_id: int) -> None:
        """Ensure crowd_cells has API top-3 when megatip cache exists."""
        from megax.crowd_observed import merge_api_top3_into_cells

        if self.megatip is None:
            return
        match = self.megatip.matches.get(str(match_id))
        if match is None or not match.top3:
            return
        self.ensure_match(match_id)
        key = str(match_id)
        existing = self.crowd_cells.get(key, {})
        overwrite = not existing
        self.crowd_cells[key] = merge_api_top3_into_cells(
            existing,
            match.top3,
            overwrite=overwrite,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_size": self.field_size,
            "round_id": self.round_id,
            "round_number": self.round_number,
            "accounts": {
                name: {
                    "rank": account.rank,
                    "points": account.points,
                    "joker_match_id": account.joker_match_id,
                    "tips": dict(account.tips),
                }
                for name, account in self.accounts.items()
            },
            "money": self.money,
            "crowd_cells": self.crowd_cells,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "last_simulation": self.last_simulation.to_dict() if self.last_simulation else None,
            "last_optimization": self.last_optimization.to_dict() if self.last_optimization else None,
            "pending_mc_job": self.pending_mc_job,
            "pending_mc_started_at": self.pending_mc_started_at,
            "megatip": self.megatip.to_dict() if self.megatip else None,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RoundGuiState:
        accounts: dict[str, AccountState] = {}
        for name in ("A", "B"):
            account_raw = (raw.get("accounts") or {}).get(name) or {}
            accounts[name] = AccountState(
                rank=_optional_int(account_raw.get("rank")),
                points=int(account_raw.get("points") or 0),
                joker_match_id=_optional_int(account_raw.get("joker_match_id")),
                tips={
                    str(k): str(v)
                    for k, v in (account_raw.get("tips") or {}).items()
                    if v
                },
            )
        calibration_raw = raw.get("calibration")
        calibration = (
            CalibrationSnapshot.from_dict(calibration_raw)
            if isinstance(calibration_raw, dict)
            else None
        )
        megatip_raw = raw.get("megatip")
        megatip = (
            MegatipGuiCache.from_dict(megatip_raw)
            if isinstance(megatip_raw, dict)
            else None
        )
        sim_raw = raw.get("last_simulation")
        last_simulation = (
            SimulationSnapshot.from_dict(sim_raw)
            if isinstance(sim_raw, dict)
            else None
        )
        opt_raw = raw.get("last_optimization")
        last_optimization = (
            OptimizerSnapshot.from_dict(opt_raw)
            if isinstance(opt_raw, dict)
            else None
        )
        return cls(
            field_size=int(raw.get("field_size") or DEFAULT_FIELD_SIZE),
            round_id=_optional_int(raw.get("round_id")),
            round_number=_optional_int(raw.get("round_number")),
            accounts=accounts,
            money={
                str(match_id): {
                    book: {k: _optional_float(values.get(k)) for k in MONEY_KEYS}
                    for book, values in (books or {}).items()
                    if book in BOOKMAKERS
                }
                for match_id, books in (raw.get("money") or {}).items()
            },
            crowd_cells=_load_crowd_cells(raw),
            calibration=calibration,
            last_simulation=last_simulation,
            last_optimization=last_optimization,
            pending_mc_job=str(raw["pending_mc_job"]) if raw.get("pending_mc_job") else None,
            pending_mc_started_at=str(raw["pending_mc_started_at"])
            if raw.get("pending_mc_started_at")
            else None,
            megatip=megatip,
        )


def _load_crowd_cells(raw: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Load crowd_cells, migrating legacy crowd_observed label keys (1:1 → 1_1)."""
    from megax.crowd_observed import label_to_cell_key

    result: dict[str, dict[str, float]] = {}
    sources = [raw.get("crowd_cells") or {}, raw.get("crowd_observed") or {}]
    for source in sources:
        for match_id, scores in source.items():
            if not isinstance(scores, dict):
                continue
            bucket = result.setdefault(str(match_id), {})
            for label, pct in scores.items():
                if pct is None:
                    continue
                key = label if "_" in str(label) and ":" not in str(label) else label_to_cell_key(str(label))
                if key is None:
                    continue
                bucket[key] = float(pct)
    return result


def gui_data_dir() -> Path:
    return PROJECT_ROOT / "data" / "gui"


def state_path(round_key: str) -> Path:
    return gui_data_dir() / f"{round_key}.json"


def load_round_state(round_key: str) -> RoundGuiState:
    from megax.storage import load_round_record

    record = load_round_record(round_key)
    if record is None:
        return RoundGuiState()
    return record.state


def save_round_state(round_key: str, state: RoundGuiState) -> None:
    """Legacy helper — prefer megax.storage.save_round_record with matches."""
    path = state_path(round_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_tip_score(text: str) -> tuple[int, int] | None:
    cleaned = text.strip().replace(" ", "")
    if not cleaned or ":" not in cleaned:
        return None
    left, right = cleaned.split(":", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None
