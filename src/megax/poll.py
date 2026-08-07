"""Poll Tipsport results for matches in an active round."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from megax.config import MegaxConfig, load_config
from megax.tipsport.client import TipsportClient
from megax.tipsport.results import (
    DEFAULT_RESULTS_POLL_DELAY,
    MatchResult,
    MatchStatus,
    poll_match_results,
)

logger = logging.getLogger(__name__)


def _results_poll_delay(config: MegaxConfig | None) -> timedelta:
    if config is None:
        return DEFAULT_RESULTS_POLL_DELAY
    return timedelta(minutes=config.results_poll_min_after_kickoff_minutes)


@dataclass(frozen=True)
class ResultsPollSnapshot:
    polled_at: datetime
    results: dict[int, MatchResult | None]


def poll_once(
    match_ids: list[int],
    *,
    kickoffs: dict[int, datetime] | None = None,
    config: MegaxConfig | None = None,
    client: TipsportClient | None = None,
    now: datetime | None = None,
) -> ResultsPollSnapshot:
    config = config or load_config()
    client = client or TipsportClient(
        config.tipsport_base_url,
        state_file=config.tipsport_state_file,
    )
    delay = _results_poll_delay(config)
    results = poll_match_results(
        client,
        match_ids,
        kickoffs=kickoffs,
        min_delay=delay,
        now=now,
    )
    polled_at = now or datetime.now(timezone.utc)
    return ResultsPollSnapshot(
        polled_at=polled_at,
        results=results,
    )


def poll_until_all_finished(
    match_ids: list[int],
    *,
    kickoffs: dict[int, datetime] | None = None,
    config: MegaxConfig | None = None,
    client: TipsportClient | None = None,
    max_iterations: int = 120,
    stop_event=None,
) -> ResultsPollSnapshot:
    config = config or load_config()
    client = client or TipsportClient(
        config.tipsport_base_url,
        state_file=config.tipsport_state_file,
    )
    latest = ResultsPollSnapshot(polled_at=datetime.now(timezone.utc), results={})
    for iteration in range(max_iterations):
        if stop_event is not None and stop_event.is_set():
            break
        latest = poll_once(match_ids, kickoffs=kickoffs, config=config, client=client)
        finished = [
            match_id
            for match_id, result in latest.results.items()
            if result is not None and result.status == MatchStatus.FINISHED
        ]
        logger.info(
            "Results poll %d/%d — finished %d/%d",
            iteration + 1,
            max_iterations,
            len(finished),
            len(match_ids),
        )
        if len(finished) == len(match_ids):
            break
        time.sleep(config.results_poll_interval_sec)
    return latest
