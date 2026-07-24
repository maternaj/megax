"""Default Megatipovačka round window (Pá–Po, Europe/Prague)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

PRAGUE = ZoneInfo("Europe/Prague")

__all__ = ["PRAGUE", "default_round_window", "round_key", "parse_day", "day_bounds"]


def default_round_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    """Return UTC bounds for the current or next Fri–Mon round window."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(PRAGUE).date()
    weekday = today.weekday()  # Mon=0 … Sun=6

    if weekday == 0:
        friday = today - timedelta(days=3)
        monday = today
    elif weekday <= 3:
        friday = today + timedelta(days=4 - weekday)
        monday = friday + timedelta(days=3)
    else:
        friday = today - timedelta(days=weekday - 4)
        monday = friday + timedelta(days=3)

    date_from = datetime.combine(friday, time.min, tzinfo=PRAGUE).astimezone(timezone.utc)
    date_to = datetime.combine(monday, time.max, tzinfo=PRAGUE).astimezone(timezone.utc)
    return date_from, date_to


def round_key(date_from: datetime, date_to: datetime) -> str:
    start = date_from.astimezone(PRAGUE).date().isoformat()
    end = date_to.astimezone(PRAGUE).date().isoformat()
    return f"{start}_{end}"


def parse_day(value: str) -> date:
    return date.fromisoformat(value)


def day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, tzinfo=PRAGUE).astimezone(timezone.utc)
    end = datetime.combine(day, time.max, tzinfo=PRAGUE).astimezone(timezone.utc)
    return start, end
