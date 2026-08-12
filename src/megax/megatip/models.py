"""Megatipovačka API models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ScoreTip:
    home: int
    away: int
    percentage: int
    is_floor: bool = False

    @property
    def label(self) -> str:
        return f"{self.home}:{self.away}"


@dataclass(frozen=True)
class PopularTips:
    top3: tuple[ScoreTip, ScoreTip, ScoreTip]
    message: str | None = None

    def as_dict(self) -> dict[str, int]:
        return {tip.label: tip.percentage for tip in self.top3}


@dataclass(frozen=True)
class ClientTip:
    round_match_id: int
    home: int
    away: int
    score: int | None = None
    joker_used: bool = False


@dataclass(frozen=True)
class RoundMatch:
    round_match_id: int
    match_id: int
    match_name: str
    kickoff_at: datetime | None
    status: str
    popular_tips: PopularTips | None
    result_home: int | None = None
    result_away: int | None = None

    @property
    def is_open(self) -> bool:
        return self.status == "MATCH_PREMATCH"


@dataclass(frozen=True)
class RoundTipsSnapshot:
    contest_id: int
    round_id: int | None
    round_matches: tuple[RoundMatch, ...]
    client_tips: tuple[ClientTip, ...] = ()
    can_tip: bool = True

    def by_match_id(self) -> dict[int, RoundMatch]:
        return {match.match_id: match for match in self.round_matches}

    def by_round_match_id(self) -> dict[int, RoundMatch]:
        return {match.round_match_id: match for match in self.round_matches}


@dataclass(frozen=True)
class PopularTipProbe:
    round_match_id: int
    queried: ScoreTip
    popular_tips: PopularTips
    client_tip: ScoreTip | None = None


@dataclass(frozen=True)
class RankingEntry:
    position: int
    score: int
    username: str | None = None
    prize: str | None = None
    is_client: bool = False


@dataclass(frozen=True)
class RankingSnapshot:
    contest_id: int
    serie_id: int | None
    round_id: int | None
    number_of_players: int
    entries: tuple[RankingEntry, ...]
    client_entry: RankingEntry | None = None
    leader_score: int | None = None


@dataclass(frozen=True)
class TileSnapshot:
    contest_id: int
    tile_id: int
    rank: int | None
    field_size: int | None
    points: int | None
    current_round: int | None
    total_rounds: int | None
    next_kickoff_text: str | None = None


@dataclass(frozen=True)
class JokerActionResult:
    joker_used: bool
    free_joker: bool
    message: str | None = None
    refresh_tip_page: bool = False


@dataclass(frozen=True)
class ObservedTipCoverage:
    """Observed crowd percentages keyed by score label."""

    match_id: int
    round_match_id: int
    tips: dict[str, int] = field(default_factory=dict)
    probed: dict[str, int] = field(default_factory=dict)
    inferred: dict[str, int] = field(default_factory=dict)
    is_floor: dict[str, bool] = field(default_factory=dict)
    failed_probes: tuple[str, ...] = ()

    def known_labels(self) -> set[str]:
        return set(self.tips) | set(self.probed) | set(self.inferred)

    def get_percentage(self, home: int, away: int) -> int | None:
        label = f"{home}:{away}"
        if label in self.tips:
            return self.tips[label]
        if label in self.probed:
            return self.probed[label]
        return self.inferred.get(label)


@dataclass(frozen=True)
class CrowdObservedSnapshot:
    round_id: int
    contest_id: int
    fetched_at: datetime
    matches: tuple[ObservedTipCoverage, ...]

    def by_match_id(self) -> dict[int, ObservedTipCoverage]:
        return {match.match_id: match for match in self.matches}
