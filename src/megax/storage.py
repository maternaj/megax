"""Local JSON persistence for Megatipovačka rounds (no Postgres)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from megax.config import PROJECT_ROOT
from megax.gui.state import RoundGuiState
from megax.team_mu import TeamOuLine
from megax.tipsport.offer import MatchOdds, MegaxMatch

SCHEMA_VERSION = 2


def rounds_data_dir() -> Path:
    return PROJECT_ROOT / "data" / "rounds"


def round_record_path(round_key: str) -> Path:
    return rounds_data_dir() / f"{round_key}.json"


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def _parse_dt(raw: object) -> datetime | None:
    if not raw:
        return None
    text = str(raw)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _team_lines_to_dict(lines: tuple[TeamOuLine, ...]) -> list[dict[str, float]]:
    return [{"line": line.line, "over": line.over, "under": line.under} for line in lines]


def _team_lines_from_dict(raw: object) -> tuple[TeamOuLine, ...]:
    if not isinstance(raw, list):
        return ()
    lines: list[TeamOuLine] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            lines.append(
                TeamOuLine(
                    line=float(item["line"]),
                    over=float(item["over"]),
                    under=float(item["under"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(lines)


def match_to_dict(match: MegaxMatch) -> dict[str, Any]:
    odds = match.odds
    return {
        "match_id": match.match_id,
        "name": match.name,
        "home": match.home,
        "away": match.away,
        "kickoff_at": _iso(match.kickoff_at),
        "match_type": match.match_type,
        "ended": match.ended,
        "competition_id": match.competition_id,
        "odds": {
            "home": odds.home,
            "draw": odds.draw,
            "away": odds.away,
            "over_2_5": odds.over_2_5,
            "under_2_5": odds.under_2_5,
            "home_team_lines": _team_lines_to_dict(odds.home_team_lines),
            "away_team_lines": _team_lines_to_dict(odds.away_team_lines),
            "match_total_lines": _team_lines_to_dict(odds.match_total_lines),
        },
    }


def match_from_dict(raw: dict[str, Any]) -> MegaxMatch | None:
    try:
        odds_raw = raw.get("odds") or {}
        kickoff = _parse_dt(raw.get("kickoff_at"))
        if kickoff is None:
            return None
        return MegaxMatch(
            match_id=int(raw["match_id"]),
            name=str(raw.get("name") or ""),
            home=str(raw.get("home") or ""),
            away=str(raw.get("away") or ""),
            kickoff_at=kickoff,
            odds=MatchOdds(
                home=float(odds_raw["home"]),
                draw=float(odds_raw["draw"]),
                away=float(odds_raw["away"]),
                over_2_5=_optional_float(odds_raw.get("over_2_5")),
                under_2_5=_optional_float(odds_raw.get("under_2_5")),
                home_team_lines=_team_lines_from_dict(odds_raw.get("home_team_lines")),
                away_team_lines=_team_lines_from_dict(odds_raw.get("away_team_lines")),
                match_total_lines=_team_lines_from_dict(odds_raw.get("match_total_lines")),
            ),
            match_type=str(raw.get("match_type") or ""),
            ended=bool(raw.get("ended")),
            competition_id=int(raw.get("competition_id") or 0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


@dataclass
class RoundRecord:
    round_key: str
    state: RoundGuiState
    matches: tuple[MegaxMatch, ...]
    saved_at: datetime
    fetched_at: datetime | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = self.state.to_dict()
        payload["schema_version"] = self.schema_version
        payload["round_key"] = self.round_key
        payload["saved_at"] = _iso(self.saved_at)
        payload["fetched_at"] = _iso(self.fetched_at)
        payload["matches"] = [match_to_dict(match) for match in self.matches]
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, round_key: str) -> RoundRecord:
        matches = tuple(
            parsed
            for item in (raw.get("matches") or [])
            if (parsed := match_from_dict(item)) is not None
        )
        return cls(
            round_key=round_key,
            state=RoundGuiState.from_dict(raw),
            matches=matches,
            saved_at=_parse_dt(raw.get("saved_at")) or datetime.now(timezone.utc),
            fetched_at=_parse_dt(raw.get("fetched_at")),
            schema_version=int(raw.get("schema_version") or 1),
        )


def load_round_record(round_key: str) -> RoundRecord | None:
    path = round_record_path(round_key)
    if not path.exists():
        legacy = PROJECT_ROOT / "data" / "gui" / f"{round_key}.json"
        if legacy.exists():
            raw = json.loads(legacy.read_text(encoding="utf-8"))
            return RoundRecord.from_dict(raw, round_key=round_key)
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RoundRecord.from_dict(raw, round_key=round_key)


def save_round_record(record: RoundRecord) -> None:
    path = round_record_path(record.round_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def list_saved_round_keys() -> list[str]:
    directory = rounds_data_dir()
    if not directory.exists():
        legacy = PROJECT_ROOT / "data" / "gui"
        if legacy.exists():
            return sorted(p.stem for p in legacy.glob("*.json"))
        return []
    return sorted(p.stem for p in directory.glob("*.json"))
