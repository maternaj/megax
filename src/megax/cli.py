"""CLI entry points for MegaX data pipelines."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, time, timezone
from typing import Any

from megax.config import load_config
from megax.ingest import fetch_round_snapshot
from megax.poll import poll_once, poll_until_all_finished
from megax.simulate import (
    SimulationConfig,
    format_simulation_report,
    load_and_simulate,
)
from megax.tipsport.client import TipsportClient
from megax.tipsport.offer import MegaxMatch
from megax.tipsport.results import MatchResult


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


def _parse_date(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _day_bounds(date_text: str) -> tuple[datetime, datetime]:
    day = _parse_date(date_text).date()
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    end = datetime.combine(day, time.max, tzinfo=timezone.utc)
    return start, end


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def _match_to_dict(match: MegaxMatch) -> dict[str, Any]:
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
            "home": match.odds.home,
            "draw": match.odds.draw,
            "away": match.odds.away,
            "over_2_5": match.odds.over_2_5,
            "under_2_5": match.odds.under_2_5,
        },
    }


def _result_to_dict(result: MatchResult) -> dict[str, Any]:
    return {
        "match_id": result.match_id,
        "status": result.status.value,
        "home_goals": result.home_goals,
        "away_goals": result.away_goals,
        "score": f"{result.home_goals}:{result.away_goals}" if result.home_goals is not None else None,
        "ended": result.ended,
        "observed_at": _iso(result.observed_at),
    }


def _round_to_dict(snapshot) -> dict[str, Any]:
    return {
        "competition_id": snapshot.competition_id,
        "date_from": _iso(snapshot.date_from),
        "date_to": _iso(snapshot.date_to),
        "fetched_at": _iso(snapshot.fetched_at),
        "match_count": len(snapshot.matches),
        "slot_count": len(snapshot.slots),
        "matches": [_match_to_dict(match) for match in snapshot.matches],
        "slots": [
            {
                "kickoff_at": _iso(slot.kickoff_at),
                "matches": [_match_to_dict(match) for match in slot.matches],
            }
            for slot in snapshot.slots
        ],
    }


def cmd_fetch_round(args: argparse.Namespace) -> int:
    if args.from_day and args.to_day:
        date_from, _ = _day_bounds(args.from_day)
        _, date_to = _day_bounds(args.to_day)
    elif args.from_date and args.to_date:
        date_from = _parse_date(args.from_date)
        date_to = _parse_date(args.to_date)
    else:
        raise SystemExit("fetch-round requires --from-date/--to-date or --from-day/--to-day")

    snapshot = fetch_round_snapshot(date_from=date_from, date_to=date_to)
    payload = _round_to_dict(snapshot)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")
    else:
        print(text)
    return 0


def cmd_poll_results(args: argparse.Namespace) -> int:
    config = load_config()
    client = TipsportClient(config.tipsport_base_url, state_file=config.tipsport_state_file)
    match_ids = args.match_id
    if args.watch:
        snapshot = poll_until_all_finished(
            match_ids,
            config=config,
            client=client,
            max_iterations=args.max_iterations,
        )
    else:
        snapshot = poll_once(match_ids, config=config, client=client)

    payload = {
        "polled_at": _iso(snapshot.polled_at),
        "results": {
            str(match_id): _result_to_dict(result) if result else None
            for match_id, result in snapshot.results.items()
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")
    else:
        print(text)
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    sim_config = SimulationConfig(
        universes=args.universes,
        field_size=args.field,
        crowd_players=args.crowd_players,
        seed=args.seed,
    )
    result = load_and_simulate(args.round, sim_config=sim_config)
    report = format_simulation_report(result)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(report)
            fh.write("\n")
    else:
        print(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="megax", description="Megatipovačka data tools")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-round", help="Fetch Czech league offer for a kickoff window")
    fetch.add_argument("--from-date", help="ISO datetime lower bound (UTC if naive)")
    fetch.add_argument("--to-date", help="ISO datetime upper bound (UTC if naive)")
    fetch.add_argument("--from-day", help="Date-only lower bound, e.g. 2026-04-04")
    fetch.add_argument("--to-day", help="Date-only upper bound, e.g. 2026-04-07")
    fetch.add_argument("-o", "--output", help="Write JSON snapshot to file")
    fetch.set_defaults(func=cmd_fetch_round)

    poll = sub.add_parser("poll-results", help="Poll Tipsport results API for match IDs")
    poll.add_argument("--match-id", type=int, action="append", required=True)
    poll.add_argument("--watch", action="store_true", help="Repeat until all matches are finished")
    poll.add_argument("--max-iterations", type=int, default=120)
    poll.add_argument("-o", "--output")
    poll.set_defaults(func=cmd_poll_results)

    simulate = sub.add_parser("simulate", help="Monte Carlo round simulation from saved snapshot")
    simulate.add_argument("--round", required=True, help="Round key, e.g. 2026-07-24_2026-07-27")
    simulate.add_argument("--universes", type=int, default=10_000)
    simulate.add_argument("--field", type=int, default=50_000, help="Field size for alpha context")
    simulate.add_argument(
        "--crowd-players",
        type=int,
        default=None,
        help="Virtual crowd per universe (default min(field, 5000))",
    )
    simulate.add_argument("--seed", type=int, default=None)
    simulate.add_argument("-o", "--output")
    simulate.set_defaults(func=cmd_simulate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    if args.command == "fetch-round":
        has_day = args.from_day and args.to_day
        has_dt = args.from_date and args.to_date
        if not has_day and not has_dt:
            parser.error("fetch-round requires --from-date/--to-date or --from-day/--to-day")

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
