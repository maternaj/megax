"""MegaX configuration from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = PROJECT_ROOT / "state"


@dataclass(frozen=True)
class MegaxConfig:
    tipsport_base_url: str
    tipsport_competition_id: int
    tipsport_state_file: str
    max_matches_per_round: int
    results_poll_interval_sec: float
    results_poll_min_after_kickoff_minutes: int
    crowd_blend_to_p: float
    crowd_tail_gamma: float
    crowd_zero_zero_delta: float
    crowd_prelec_alpha: float
    crowd_zero_zero_min: float
    gpp_alpha: float | None
    swap_delta_small: float
    swap_delta_large: float
    swap_leader_chalk: float
    swap_chase_alpha_boost: float
    swap_protect_ev_ratio: float
    swap_chase_ev_ratio: float


def load_config(*, env_file: str | Path | None = None) -> MegaxConfig:
    if env_file is not None:
        load_dotenv(env_file)
    else:
        load_dotenv(PROJECT_ROOT / ".env")

    state_dir = os.getenv("MEGAX_STATE_DIR", str(DEFAULT_STATE_DIR))
    state_file = os.getenv(
        "MEGAX_TIPSPORT_STATE_FILE",
        str(Path(state_dir) / "tipsport_scraper_state.json"),
    )
    return MegaxConfig(
        tipsport_base_url=os.getenv("MEGAX_TIPSPORT_BASE_URL", "https://www.tipsport.cz").rstrip("/"),
        tipsport_competition_id=int(os.getenv("MEGAX_TIPSPORT_COMPETITION_ID", "120")),
        tipsport_state_file=state_file,
        max_matches_per_round=int(os.getenv("MEGAX_MAX_MATCHES_PER_ROUND", "10")),
        results_poll_interval_sec=float(os.getenv("MEGAX_RESULTS_POLL_INTERVAL_SEC", "60")),
        results_poll_min_after_kickoff_minutes=int(
            os.getenv("MEGAX_RESULTS_POLL_MIN_AFTER_KICKOFF_MIN", "105"),
        ),
        crowd_blend_to_p=float(os.getenv("MEGAX_CROWD_BLEND_TO_P", "0.30")),
        crowd_tail_gamma=float(os.getenv("MEGAX_CROWD_TAIL_GAMMA", "0.50")),
        crowd_zero_zero_delta=float(os.getenv("MEGAX_CROWD_ZERO_ZERO_DELTA", "0.20")),
        crowd_prelec_alpha=float(os.getenv("MEGAX_CROWD_PRELEC_ALPHA", "1.15")),
        crowd_zero_zero_min=float(os.getenv("MEGAX_CROWD_ZERO_ZERO_MIN", "0.015")),
        gpp_alpha=_optional_float(os.getenv("MEGAX_GPP_ALPHA")),
        swap_delta_small=float(os.getenv("MEGAX_SWAP_DELTA_SMALL", "3")),
        swap_delta_large=float(os.getenv("MEGAX_SWAP_DELTA_LARGE", "8")),
        swap_leader_chalk=float(os.getenv("MEGAX_SWAP_LEADER_CHALK", "0.85")),
        swap_chase_alpha_boost=float(os.getenv("MEGAX_SWAP_CHASE_ALPHA_BOOST", "0.3")),
        swap_protect_ev_ratio=float(os.getenv("MEGAX_SWAP_PROTECT_EV_RATIO", "0.95")),
        swap_chase_ev_ratio=float(os.getenv("MEGAX_SWAP_CHASE_EV_RATIO", "0.85")),
    )


def _optional_float(raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return None
    return float(raw)
