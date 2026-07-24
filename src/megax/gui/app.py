"""FastAPI GUI for MegaX."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from megax.config import load_config
from megax.gui.render import render_page
from megax.gui.service import build_round_view, snapshot_from_record
from megax.gui.state import (
    BOOKMAKERS,
    MONEY_KEYS,
    RoundGuiState,
)
from megax.gui.weekend import PRAGUE, day_bounds, default_round_window, parse_day, round_key
from megax.ingest import fetch_round_snapshot
from megax.lineup import apply_lineup_to_state
from megax.swap import apply_swap_to_state
from megax.poll import poll_once
from megax.storage import RoundRecord, load_round_record, save_round_record
from megax.tipsport.client import TipsportClient

logger = logging.getLogger(__name__)

APP_TITLE = "MegaX"
DEFAULT_HOST = os.getenv("MEGAX_GUI_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MEGAX_GUI_PORT", "18555"))


def _tipsport_client() -> TipsportClient:
    config = load_config()
    return TipsportClient(config.tipsport_base_url, state_file=config.tipsport_state_file)


def _window_from_form(from_day: str | None, to_day: str | None) -> tuple[datetime, datetime, str, str]:
    if from_day and to_day:
        date_from, _ = day_bounds(parse_day(from_day))
        _, date_to = day_bounds(parse_day(to_day))
        return date_from, date_to, from_day, to_day
    date_from, date_to = default_round_window()
    return date_from, date_to, _day_str(date_from), _day_str(date_to)


def _day_str(dt: datetime) -> str:
    return dt.astimezone(PRAGUE).date().isoformat()


def _is_past_round(date_to: datetime) -> bool:
    today = datetime.now(PRAGUE).date()
    return date_to.astimezone(PRAGUE).date() < today


def _apply_form_to_state(
    state: RoundGuiState,
    *,
    form: dict[str, str],
    match_ids: list[int],
) -> None:
    state.field_size = int(form.get("field_size") or state.field_size)
    for name in ("A", "B"):
        account = state.accounts[name]
        suffix = name.lower()
        account.rank = _optional_int(form.get(f"rank_{suffix}"))
        account.points = int(form.get(f"points_{suffix}") or 0)
        account.joker_match_id = _optional_int(form.get(f"joker_{suffix}"))
        account.tips = {}
        for match_id in match_ids:
            key = str(match_id)
            tip = (form.get(f"tip_{suffix}_{key}") or "").strip()
            if tip:
                account.tips[key] = tip

    for match_id in match_ids:
        state.ensure_match(match_id)
        key = str(match_id)
        for book in BOOKMAKERS:
            for money_key in MONEY_KEYS:
                field = f"money_{key}_{book}_{money_key}"
                state.money[key][book][money_key] = _optional_float(form.get(field))


def _optional_int(value: str | None) -> int | None:
    if not value:
        return None
    return int(value)


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _resolve_round(
    *,
    date_from: datetime,
    date_to: datetime,
    key: str,
    client: TipsportClient,
) -> tuple[RoundGuiState, object, bool, datetime | None]:
    record = load_round_record(key)
    state = record.state if record else RoundGuiState()
    past = _is_past_round(date_to)

    if past and record and record.matches:
        snapshot = snapshot_from_record(record, date_from=date_from, date_to=date_to)
        return state, snapshot, True, record.saved_at

    snapshot = fetch_round_snapshot(date_from=date_from, date_to=date_to, client=client)
    return state, snapshot, False, record.saved_at if record else None


def _persist_round(
    *,
    key: str,
    state: RoundGuiState,
    snapshot,
) -> None:
    save_round_record(
        RoundRecord(
            round_key=key,
            state=state,
            matches=tuple(snapshot.matches),
            saved_at=datetime.now(timezone.utc),
            fetched_at=snapshot.fetched_at,
        )
    )


def create_app() -> FastAPI:
    app = FastAPI(title=APP_TITLE, version="0.0.1")

    @app.get("/", response_class=HTMLResponse)
    async def home(
        request: Request,
        from_day: str | None = None,
        to_day: str | None = None,
        saved: int | None = None,
        filled: int | None = None,
        swapped: int | None = None,
    ) -> HTMLResponse:
        date_from, date_to, from_day_str, to_day_str = _window_from_form(from_day, to_day)
        key = round_key(date_from, date_to)
        client = _tipsport_client()
        state, snapshot, read_only, saved_at = _resolve_round(
            date_from=date_from,
            date_to=date_to,
            key=key,
            client=client,
        )
        view = build_round_view(
            date_from=date_from,
            date_to=date_to,
            round_key=key,
            state=state,
            snapshot=snapshot,
            client=client,
            read_only=read_only,
            saved_at=saved_at,
        )
        message = "Kolo uloženo (tipy + poslední kurzy)." if saved else None
        if filled:
            message = "Tipy A/B a žolíky vyplněny z optimizeru."
        if swapped:
            message = "Late swap tipy aplikovány pro zbývající sloty."
        html = render_page(view, message=message)
        return HTMLResponse(html)

    @app.post("/save")
    async def save(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        from_day = form.get("from_day")
        to_day = form.get("to_day")
        date_from, date_to, from_day_str, to_day_str = _window_from_form(from_day, to_day)
        key = round_key(date_from, date_to)
        client = _tipsport_client()
        state, snapshot, _read_only, _saved_at = _resolve_round(
            date_from=date_from,
            date_to=date_to,
            key=key,
            client=client,
        )
        match_ids = [match.match_id for match in snapshot.matches]
        _apply_form_to_state(state, form=form, match_ids=match_ids)
        _persist_round(key=key, state=state, snapshot=snapshot)
        return RedirectResponse(
            url=f"/?from_day={from_day_str}&to_day={to_day_str}&saved=1",
            status_code=303,
        )

    @app.post("/refresh")
    async def refresh(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        from_day = form.get("from_day")
        to_day = form.get("to_day")
        date_from, date_to, from_day_str, to_day_str = _window_from_form(from_day, to_day)
        key = round_key(date_from, date_to)
        client = _tipsport_client()
        state, _old_snapshot, read_only, _saved_at = _resolve_round(
            date_from=date_from,
            date_to=date_to,
            key=key,
            client=client,
        )
        if read_only:
            return RedirectResponse(
                url=f"/?from_day={from_day_str}&to_day={to_day_str}",
                status_code=303,
            )
        snapshot = fetch_round_snapshot(date_from=date_from, date_to=date_to, client=client)
        match_ids = [match.match_id for match in snapshot.matches]
        _apply_form_to_state(state, form=form, match_ids=match_ids)
        _persist_round(key=key, state=state, snapshot=snapshot)
        if match_ids:
            kickoffs = {match.match_id: match.kickoff_at for match in snapshot.matches}
            poll_once(match_ids, kickoffs=kickoffs, client=client)
        return RedirectResponse(
            url=f"/?from_day={from_day_str}&to_day={to_day_str}",
            status_code=303,
        )

    @app.post("/fill-lineup")
    async def fill_lineup(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        from_day = form.get("from_day")
        to_day = form.get("to_day")
        date_from, date_to, from_day_str, to_day_str = _window_from_form(from_day, to_day)
        key = round_key(date_from, date_to)
        client = _tipsport_client()
        state, snapshot, read_only, _saved_at = _resolve_round(
            date_from=date_from,
            date_to=date_to,
            key=key,
            client=client,
        )
        if read_only:
            return RedirectResponse(
                url=f"/?from_day={from_day_str}&to_day={to_day_str}",
                status_code=303,
            )
        match_ids = [match.match_id for match in snapshot.matches]
        _apply_form_to_state(state, form=form, match_ids=match_ids)
        view = build_round_view(
            date_from=date_from,
            date_to=date_to,
            round_key=key,
            state=state,
            snapshot=snapshot,
            client=client,
        )
        if view.lineup is not None:
            apply_lineup_to_state(state, view.lineup)
        _persist_round(key=key, state=state, snapshot=snapshot)
        return RedirectResponse(
            url=f"/?from_day={from_day_str}&to_day={to_day_str}&filled=1",
            status_code=303,
        )

    @app.post("/apply-swap")
    async def apply_swap(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        from_day = form.get("from_day")
        to_day = form.get("to_day")
        date_from, date_to, from_day_str, to_day_str = _window_from_form(from_day, to_day)
        key = round_key(date_from, date_to)
        client = _tipsport_client()
        state, snapshot, read_only, _saved_at = _resolve_round(
            date_from=date_from,
            date_to=date_to,
            key=key,
            client=client,
        )
        if read_only:
            return RedirectResponse(
                url=f"/?from_day={from_day_str}&to_day={to_day_str}",
                status_code=303,
            )
        match_ids = [match.match_id for match in snapshot.matches]
        _apply_form_to_state(state, form=form, match_ids=match_ids)
        view = build_round_view(
            date_from=date_from,
            date_to=date_to,
            round_key=key,
            state=state,
            snapshot=snapshot,
            client=client,
        )
        if view.swap is not None:
            apply_swap_to_state(
                state,
                view.swap,
                remaining_match_ids=set(view.swap.remaining_match_ids),
            )
        _persist_round(key=key, state=state, snapshot=snapshot)
        return RedirectResponse(
            url=f"/?from_day={from_day_str}&to_day={to_day_str}&swapped=1",
            status_code=303,
        )

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MegaX Megatipovačka GUI")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--log-level", default=os.getenv("MEGAX_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
