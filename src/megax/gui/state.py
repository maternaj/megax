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
    accounts: dict[str, AccountState] = field(default_factory=lambda: {
        "A": AccountState(),
        "B": AccountState(),
    })
    money: dict[str, dict[str, dict[str, float | None]]] = field(default_factory=dict)
    calibration: CalibrationSnapshot | None = None

    def ensure_match(self, match_id: int) -> None:
        key = str(match_id)
        if key not in self.money:
            self.money[key] = {
                book: {k: None for k in MONEY_KEYS}
                for book in BOOKMAKERS
            }

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_size": self.field_size,
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
            "calibration": self.calibration.to_dict() if self.calibration else None,
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
        return cls(
            field_size=int(raw.get("field_size") or DEFAULT_FIELD_SIZE),
            accounts=accounts,
            money={
                str(match_id): {
                    book: {k: _optional_float(values.get(k)) for k in MONEY_KEYS}
                    for book, values in (books or {}).items()
                    if book in BOOKMAKERS
                }
                for match_id, books in (raw.get("money") or {}).items()
            },
            calibration=calibration,
        )


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
