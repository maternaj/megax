"""FastAPI GUI for MegaX."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from megax.calibrate import (
    build_lineup_for_knobs,
    calibration_snapshot_from_result,
    gui_calibration_config,
    knobs_from_snapshot,
    load_and_calibrate,
)
from megax.config import load_config
from megax.gui.jobs import job_store, recover_job_payload
from megax.gui.megatip_bridge import apply_megatip_to_state, fetch_megatip_round, megatip_api
from megax.crowd_observed import CROWD_GRID_SIZE, merge_api_top3_into_cells
from megax.gui.render import render_page
from megax.gui.service import build_round_view, snapshot_from_record
from megax.gui.state import (
    RoundGuiState,
    parse_tip_score,
)
from megax.gui.weekend import PRAGUE, day_bounds, default_round_window, parse_day, round_key
from megax.ingest import fetch_round_snapshot
from megax.lineup import apply_lineup_to_state
from megax.megatip.round import detect_current_round_id
from megax.poll import poll_once
from megax.optimize import (
    apply_optimizer_snapshot,
    estimate_optimize_units,
    gui_optimize_config_pair,
    optimize_round_record,
    optimizer_snapshot_from_result,
)
from megax.simulate import (
    build_match_sim_contexts,
    gui_simulation_config,
    simulate_round_record,
    simulation_snapshot_from_result,
)
from megax.storage import RoundRecord, load_round_record, round_storage_key, save_round_record
from megax.swap import apply_swap_to_state
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
    if form.get("round_id"):
        state.round_id = int(form["round_id"])
    if form.get("round_number"):
        state.round_number = int(form["round_number"])
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
        _apply_crowd_cells_from_form(state, str(match_id), form)


def _apply_crowd_cells_from_form(
    state: RoundGuiState,
    match_key: str,
    form: dict[str, str],
    *,
    grid_size: int = CROWD_GRID_SIZE,
) -> None:
    cells: dict[str, float] = {}
    for home in range(grid_size):
        for away in range(grid_size):
            field = f"crowd_{match_key}_{home}_{away}"
            raw = (form.get(field) or "").strip()
            if raw == "":
                continue
            cells[f"{home}_{away}"] = float(raw)
    state.crowd_cells[match_key] = cells


def _optional_int(value: str | None) -> int | None:
    if not value:
        return None
    return int(value)


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _redirect_round(round_id: int, *, extra: str = "") -> RedirectResponse:
    suffix = f"&{extra}" if extra else ""
    return RedirectResponse(url=f"/?round_id={round_id}{suffix}", status_code=303)


def _resolve_legacy_round(
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


def _resolve_round_by_id(
    round_id: int,
    *,
    client: TipsportClient,
    refresh_megatip: bool = False,
) -> tuple[RoundGuiState, object, datetime | None, str | None]:
    key = round_storage_key(round_id)
    record = load_round_record(key)
    state = record.state if record else RoundGuiState(round_id=round_id)
    state.round_id = round_id

    if record and record.matches and not refresh_megatip:
        snapshot = snapshot_from_record(
            record,
            date_from=record.matches[0].kickoff_at,
            date_to=record.matches[-1].kickoff_at,
        )
        return state, snapshot, record.saved_at, None

    try:
        result = fetch_megatip_round(round_id, tipsport_client=client)
    except Exception as exc:
        logger.exception("Megatip fetch failed for round %s", round_id)
        return state, None, record.saved_at if record else None, str(exc)

    if result is None:
        return state, None, record.saved_at if record else None, "Megatip API nevrátilo data pro toto kolo."

    apply_megatip_to_state(state, result)
    return state, result.snapshot, record.saved_at if record else None, None


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


def _mark_pending_mc_job(key: str, *, kind: str) -> None:
    record = load_round_record(key)
    if record is None:
        return
    record.state.pending_mc_job = kind
    record.state.pending_mc_started_at = datetime.now(timezone.utc).isoformat()
    save_round_record(record)


def _clear_pending_mc_job(key: str) -> None:
    record = load_round_record(key)
    if record is None:
        return
    record.state.pending_mc_job = None
    record.state.pending_mc_started_at = None
    save_round_record(record)


def _redirect_after_job(
    round_id: int | None,
    *,
    from_day_str: str,
    to_day_str: str,
    extra: str,
) -> RedirectResponse:
    if round_id is not None:
        return _redirect_round(round_id, extra=extra)
    return RedirectResponse(
        url=f"/?from_day={from_day_str}&to_day={to_day_str}&{extra}",
        status_code=303,
    )


def _run_optimize_job(
    *,
    key: str,
    metric: str,
    search_cfg,
    final_cfg,
    universes: int,
    crowd_players: int,
    field_size: int,
) -> None:
    from megax.gui.state import OptimizerSnapshot

    def on_progress(_phase: str, done: float, total: float, detail: str) -> None:
        job_store.update(key, phase=_phase, message=detail, done=done, total=total)

    try:
        record = load_round_record(key)
        if record is None:
            raise RuntimeError(f"Round snapshot missing: {key}")
        result = optimize_round_record(
            record,
            metric=metric,  # type: ignore[arg-type]
            search_config=search_cfg,
            final_config=final_cfg,
            progress=on_progress,
        )
        record = load_round_record(key)
        if record is None:
            raise RuntimeError(f"Round snapshot missing after optimize: {key}")
        record.state.last_optimization = optimizer_snapshot_from_result(result)
        save_round_record(record)
        _clear_pending_mc_job(key)
        job_store.complete(key, message=result.note or "Optimalizace dokončena")
    except Exception as exc:
        logger.exception("Background optimization failed for %s", key)
        record = load_round_record(key)
        if record is not None:
            record.state.last_optimization = OptimizerSnapshot(
                metric=metric,
                objective=0.0,
                tips_a={},
                tips_b={},
                joker_a=0,
                joker_b=0,
                p_win_a=0.0,
                p_top_10_a=0.0,
                p_top_100_a=0.0,
                mean_pts_a=0.0,
                p_win_b=0.0,
                p_top_10_b=0.0,
                p_top_100_b=0.0,
                mean_pts_b=0.0,
                universes=universes,
                crowd_players=crowd_players,
                field_size=field_size,
                optimized_at=datetime.now(timezone.utc).isoformat(),
                error=str(exc),
            )
            save_round_record(record)
        _clear_pending_mc_job(key)
        job_store.fail(key, str(exc))


def _run_simulate_job(
    *,
    key: str,
    sim_config,
    universes: int,
    crowd_players: int,
    field_size: int,
) -> None:
    from megax.gui.state import SimulationSnapshot

    def on_progress(done: int, total: int) -> None:
        job_store.update(
            key,
            phase="simulate",
            message=f"Simulace · universe {done:,}/{total:,}",
            done=float(done),
            total=float(total),
        )

    try:
        record = load_round_record(key)
        if record is None:
            raise RuntimeError(f"Round snapshot missing: {key}")
        result = simulate_round_record(record, sim_config=sim_config, progress=on_progress)
        record = load_round_record(key)
        if record is None:
            raise RuntimeError(f"Round snapshot missing after simulate: {key}")
        record.state.last_simulation = simulation_snapshot_from_result(result)
        save_round_record(record)
        _clear_pending_mc_job(key)
        job_store.complete(key, message="Simulace dokončena")
    except Exception as exc:
        logger.exception("Background simulation failed for %s", key)
        record = load_round_record(key)
        if record is not None:
            record.state.last_simulation = SimulationSnapshot(
                universes=universes,
                crowd_players=crowd_players,
                field_size=field_size,
                simulated_at=datetime.now(timezone.utc).isoformat(),
                agents=(),
                error=str(exc),
            )
            save_round_record(record)
        _clear_pending_mc_job(key)
        job_store.fail(key, str(exc))


def create_app() -> FastAPI:
    app = FastAPI(title=APP_TITLE, version="0.0.2")

    @app.get("/api/job-status")
    async def job_status_route(round_key: str, expect: str | None = None) -> JSONResponse:
        job = job_store.get(round_key)
        if job is not None:
            return JSONResponse(job.to_dict())
        if expect in ("optimize", "simulate"):
            recovered = recover_job_payload(round_key, expect)  # type: ignore[arg-type]
            if recovered is not None:
                return JSONResponse(recovered)
        return JSONResponse({"running": False, "status": "idle"})

    @app.get("/", response_class=HTMLResponse)
    async def home(
        request: Request,
        round_id: int | None = None,
        from_day: str | None = None,
        to_day: str | None = None,
        saved: int | None = None,
        filled: int | None = None,
        swapped: int | None = None,
        calibrated: int | None = None,
        sim: int | None = None,
        optimizing: int | None = None,
        simulating: int | None = None,
        optimized: int | None = None,
        megatip: int | None = None,
        error: str | None = None,
    ) -> HTMLResponse:
        client = _tipsport_client()
        config = load_config()
        megatip_error = error
        read_only = False
        saved_at = None

        if round_id is None and from_day and to_day:
            date_from, date_to, from_day_str, to_day_str = _window_from_form(from_day, to_day)
            key = round_key(date_from, date_to)
            state, snapshot, read_only, saved_at = _resolve_legacy_round(
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
        else:
            if round_id is None:
                api = megatip_api(config)
                round_id = detect_current_round_id(
                    api,
                    offset=config.megatip_round_id_offset,
                )

            if round_id is None:
                html = render_page(
                    None,
                    message=megatip_error,
                    error="Zadejte round_id v URL nebo formuláři (auto-detekce z veřejného API selhala).",
                )
                return HTMLResponse(html)

            key = round_storage_key(round_id)
            state, snapshot, saved_at, fetch_error = _resolve_round_by_id(
                round_id,
                client=client,
            )
            if fetch_error and megatip_error is None:
                megatip_error = fetch_error
            if snapshot is None:
                html = render_page(
                    None,
                    round_id=round_id,
                    message=megatip_error,
                    error="Nepodařilo se načíst kolo — zkuste Obnovit top-3.",
                )
                return HTMLResponse(html)

            view = build_round_view(
                date_from=snapshot.date_from,
                date_to=snapshot.date_to,
                round_key=key,
                state=state,
                snapshot=snapshot,
                client=client,
                read_only=read_only,
                saved_at=saved_at,
                round_id=round_id,
            )

        message = "Kolo uloženo (tipy + poslední kurzy)." if saved else None
        if filled:
            message = "Tipy A/B a žolíky vyplněny z optimizeru."
        if swapped:
            message = "Late swap tipy aplikovány pro zbývající sloty."
        if calibrated:
            message = "Kalibrace dokončena — tipy A/B vyplněny doporučenými knoby."
        if sim:
            message = "Monte Carlo simulace dokončena — viz tabulka níže."
        if optimized:
            message = "MC optimalizace tipů dokončena — viz výsledek nebo Vyplnit Tip A/B."
        if megatip:
            message = "Veřejná Megatip data načtena (top-3 tipy davu)."
        if error == "job_running":
            message = "MC běh už probíhá — počkejte na progress lištu."
        elif error == "no_p_matrix":
            message = "Optimalizace vyžaduje alespoň jeden zápas s maticí P."
        elif megatip_error and not message:
            message = megatip_error

        render_error = megatip_error if snapshot is None else None
        if render_error in ("job_running", "no_p_matrix"):
            render_error = None

        job_expect: str | None = None
        if optimizing:
            job_expect = "optimize"
        elif simulating:
            job_expect = "simulate"
        elif job_store.is_running(view.round_key):
            running = job_store.get(view.round_key)
            job_expect = running.kind if running is not None else None

        html = render_page(
            view,
            message=message,
            error=render_error,
            poll_job=bool(optimizing or simulating or job_store.is_running(view.round_key)),
            job_expect=job_expect,
        )
        return HTMLResponse(html)

    def _round_context_from_form(form: dict[str, str]) -> tuple[int | None, str | None, datetime, datetime, str, str]:
        if form.get("round_id"):
            round_id = int(form["round_id"])
            key = round_storage_key(round_id)
            record = load_round_record(key)
            if record and record.matches:
                snapshot = snapshot_from_record(
                    record,
                    date_from=record.matches[0].kickoff_at,
                    date_to=record.matches[-1].kickoff_at,
                )
                return (
                    round_id,
                    key,
                    snapshot.date_from,
                    snapshot.date_to,
                    _day_str(snapshot.date_from),
                    _day_str(snapshot.date_to),
                )
            result = fetch_megatip_round(round_id, tipsport_client=_tipsport_client())
            if result is None:
                raise RuntimeError(f"Round {round_id} unavailable")
            return (
                round_id,
                key,
                result.snapshot.date_from,
                result.snapshot.date_to,
                _day_str(result.snapshot.date_from),
                _day_str(result.snapshot.date_to),
            )

        from_day = form.get("from_day")
        to_day = form.get("to_day")
        date_from, date_to, from_day_str, to_day_str = _window_from_form(from_day, to_day)
        return None, round_key(date_from, date_to), date_from, date_to, from_day_str, to_day_str

    @app.post("/save")
    async def save(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        round_id, key, date_from, date_to, from_day_str, to_day_str = _round_context_from_form(form)
        client = _tipsport_client()
        if round_id is not None:
            state, snapshot, _saved_at, _err = _resolve_round_by_id(round_id, client=client)
        else:
            state, snapshot, _read_only, _saved_at = _resolve_legacy_round(
                date_from=date_from,
                date_to=date_to,
                key=key,
                client=client,
            )
        match_ids = [match.match_id for match in snapshot.matches]
        _apply_form_to_state(state, form=form, match_ids=match_ids)
        _persist_round(key=key, state=state, snapshot=snapshot)
        if round_id is not None:
            return _redirect_round(round_id, extra="saved=1")
        return RedirectResponse(
            url=f"/?from_day={from_day_str}&to_day={to_day_str}&saved=1",
            status_code=303,
        )

    @app.post("/fetch-megatip")
    async def fetch_megatip_route(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        round_id_raw = form.get("round_id")
        if not round_id_raw:
            raise RuntimeError("round_id required")
        round_id = int(round_id_raw)
        client = _tipsport_client()
        state, snapshot, _saved_at, error = _resolve_round_by_id(
            round_id,
            client=client,
            refresh_megatip=True,
        )
        if snapshot is None:
            return _redirect_round(round_id, extra=f"error={error or 'fetch_failed'}")
        match_ids = [match.match_id for match in snapshot.matches]
        _apply_form_to_state(state, form=form, match_ids=match_ids)
        _persist_round(key=round_storage_key(round_id), state=state, snapshot=snapshot)
        return _redirect_round(round_id, extra="megatip=1")

    @app.post("/reset-crowd-match")
    async def reset_crowd_match(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        round_id_raw = form.get("round_id")
        if not round_id_raw:
            raise RuntimeError("round_id required")
        round_id = int(round_id_raw)
        match_id = int(form["reset_match_id"])
        client = _tipsport_client()
        state, snapshot, _saved_at, error = _resolve_round_by_id(round_id, client=client)
        if snapshot is None:
            return _redirect_round(round_id, extra=f"error={error or 'reset_failed'}")
        match_ids = [match.match_id for match in snapshot.matches]
        _apply_form_to_state(state, form=form, match_ids=match_ids)
        if state.megatip is not None:
            mt = state.megatip.matches.get(str(match_id))
            if mt is not None and mt.top3:
                state.crowd_cells[str(match_id)] = merge_api_top3_into_cells(
                    {},
                    mt.top3,
                    overwrite=True,
                )
        _persist_round(key=round_storage_key(round_id), state=state, snapshot=snapshot)
        return _redirect_round(round_id, extra="crowd_reset=1")

    @app.post("/simulate")
    async def simulate_route(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        round_id, key, date_from, date_to, from_day_str, to_day_str = _round_context_from_form(form)
        client = _tipsport_client()
        read_only = False
        if round_id is not None:
            state, snapshot, _saved_at, _err = _resolve_round_by_id(round_id, client=client)
        else:
            state, snapshot, read_only, _saved_at = _resolve_legacy_round(
                date_from=date_from,
                date_to=date_to,
                key=key,
                client=client,
            )
        if read_only or snapshot is None:
            if round_id is not None:
                return _redirect_round(round_id, extra="error=simulate_readonly")
            return RedirectResponse(
                url=f"/?from_day={from_day_str}&to_day={to_day_str}&error=simulate_readonly",
                status_code=303,
            )
        match_ids = [match.match_id for match in snapshot.matches]
        _apply_form_to_state(state, form=form, match_ids=match_ids)
        _persist_round(key=key, state=state, snapshot=snapshot)

        universes = int(form.get("sim_universes") or 2000)
        crowd_players = int(form.get("sim_crowd_players") or 400)
        sim_config = gui_simulation_config(
            state.field_size,
            universes=universes,
            crowd_players=crowd_players,
        )
        record = load_round_record(key)
        if record is None:
            raise RuntimeError(f"Round snapshot missing after persist: {key}")

        if job_store.is_running(key):
            if round_id is not None:
                return _redirect_round(round_id, extra="error=job_running")
            return RedirectResponse(
                url=f"/?from_day={from_day_str}&to_day={to_day_str}&error=job_running",
                status_code=303,
            )

        try:
            job_store.start(
                key,
                kind="simulate",
                total=float(universes),
                message="Simulace startuje…",
                redirect_extra="sim=1",
            )
        except RuntimeError:
            if round_id is not None:
                return _redirect_round(round_id, extra="error=job_running")
            return RedirectResponse(
                url=f"/?from_day={from_day_str}&to_day={to_day_str}&error=job_running",
                status_code=303,
            )

        _mark_pending_mc_job(key, kind="simulate")

        threading.Thread(
            target=_run_simulate_job,
            kwargs={
                "key": key,
                "sim_config": sim_config,
                "universes": universes,
                "crowd_players": crowd_players,
                "field_size": state.field_size,
            },
            daemon=True,
        ).start()
        return _redirect_after_job(
            round_id,
            from_day_str=from_day_str,
            to_day_str=to_day_str,
            extra="simulating=1",
        )

    @app.post("/optimize-lineup")
    async def optimize_lineup_route(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        round_id, key, date_from, date_to, from_day_str, to_day_str = _round_context_from_form(form)
        client = _tipsport_client()
        read_only = False
        if round_id is not None:
            state, snapshot, _saved_at, _err = _resolve_round_by_id(round_id, client=client)
        else:
            state, snapshot, read_only, _saved_at = _resolve_legacy_round(
                date_from=date_from,
                date_to=date_to,
                key=key,
                client=client,
            )
        if read_only or snapshot is None:
            if round_id is not None:
                return _redirect_round(round_id, extra="error=optimize_readonly")
            return RedirectResponse(
                url=f"/?from_day={from_day_str}&to_day={to_day_str}&error=optimize_readonly",
                status_code=303,
            )
        match_ids = [match.match_id for match in snapshot.matches]
        _apply_form_to_state(state, form=form, match_ids=match_ids)
        _persist_round(key=key, state=state, snapshot=snapshot)

        metric_raw = str(form.get("opt_metric") or "top10")
        metric = metric_raw if metric_raw in ("top10", "top1", "win") else "top10"
        universes = int(form.get("sim_universes") or 2000)
        crowd_players = int(form.get("sim_crowd_players") or 400)

        search_cfg, final_cfg = gui_optimize_config_pair(
            state.field_size,
            universes=universes,
            crowd_players=crowd_players,
        )
        sim_cfg = final_cfg
        record = load_round_record(key)
        if record is None:
            raise RuntimeError(f"Round snapshot missing after persist: {key}")

        if job_store.is_running(key):
            return _redirect_after_job(
                round_id,
                from_day_str=from_day_str,
                to_day_str=to_day_str,
                extra="error=job_running",
            )

        contexts = build_match_sim_contexts(record.matches, record.state)
        if not contexts:
            return _redirect_after_job(
                round_id,
                from_day_str=from_day_str,
                to_day_str=to_day_str,
                extra="error=no_p_matrix",
            )

        _eval_budget, progress_units = estimate_optimize_units(
            contexts,
            universes=sim_cfg.universes,
        )
        try:
            job_store.start(
                key,
                kind="optimize",
                total=progress_units,
                message="Optimalizace startuje…",
                redirect_extra="optimized=1",
            )
        except RuntimeError:
            return _redirect_after_job(
                round_id,
                from_day_str=from_day_str,
                to_day_str=to_day_str,
                extra="error=job_running",
            )

        _mark_pending_mc_job(key, kind="optimize")

        threading.Thread(
            target=_run_optimize_job,
            kwargs={
                "key": key,
                "metric": metric,
                "search_cfg": sim_cfg,
                "final_cfg": sim_cfg,
                "universes": universes,
                "crowd_players": crowd_players,
                "field_size": state.field_size,
            },
            daemon=True,
        ).start()
        return _redirect_after_job(
            round_id,
            from_day_str=from_day_str,
            to_day_str=to_day_str,
            extra="optimizing=1",
        )

    @app.post("/apply-optimized-lineup")
    async def apply_optimized_lineup_route(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        round_id, key, date_from, date_to, from_day_str, to_day_str = _round_context_from_form(form)
        client = _tipsport_client()
        read_only = False
        if round_id is not None:
            state, snapshot, _saved_at, _err = _resolve_round_by_id(round_id, client=client)
        else:
            state, snapshot, read_only, _saved_at = _resolve_legacy_round(
                date_from=date_from,
                date_to=date_to,
                key=key,
                client=client,
            )
        if read_only or snapshot is None:
            if round_id is not None:
                return _redirect_round(round_id, extra="error=apply_opt_readonly")
            return RedirectResponse(
                url=f"/?from_day={from_day_str}&to_day={to_day_str}&error=apply_opt_readonly",
                status_code=303,
            )
        match_ids = [match.match_id for match in snapshot.matches]
        _apply_form_to_state(state, form=form, match_ids=match_ids)
        opt = state.last_optimization
        if opt is None or opt.error or not opt.tips_a:
            if round_id is not None:
                return _redirect_round(round_id, extra="error=no_optimization")
            return RedirectResponse(
                url=f"/?from_day={from_day_str}&to_day={to_day_str}&error=no_optimization",
                status_code=303,
            )
        apply_optimizer_snapshot(state, opt)
        _persist_round(key=key, state=state, snapshot=snapshot)
        if round_id is not None:
            return _redirect_round(round_id, extra="filled=1")
        return RedirectResponse(
            url=f"/?from_day={from_day_str}&to_day={to_day_str}&filled=1",
            status_code=303,
        )

    @app.post("/refresh")
    async def refresh(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        round_id, key, date_from, date_to, from_day_str, to_day_str = _round_context_from_form(form)
        client = _tipsport_client()
        if round_id is not None:
            state, snapshot, _saved_at, _err = _resolve_round_by_id(round_id, client=client)
        else:
            state, _old_snapshot, read_only, _saved_at = _resolve_legacy_round(
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
        if round_id is not None:
            return _redirect_round(round_id)
        return RedirectResponse(
            url=f"/?from_day={from_day_str}&to_day={to_day_str}",
            status_code=303,
        )

    def _handle_lineup_action(form: dict[str, str], *, redirect_flag: str) -> RedirectResponse:
        round_id, key, date_from, date_to, from_day_str, to_day_str = _round_context_from_form(form)
        client = _tipsport_client()
        read_only = False
        if round_id is not None:
            state, snapshot, _saved_at, _err = _resolve_round_by_id(round_id, client=client)
        else:
            state, snapshot, read_only, _saved_at = _resolve_legacy_round(
                date_from=date_from,
                date_to=date_to,
                key=key,
                client=client,
            )
        if read_only:
            if round_id is not None:
                return _redirect_round(round_id)
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
            round_id=round_id,
        )
        if view.lineup is not None:
            apply_lineup_to_state(state, view.lineup)
        _persist_round(key=key, state=state, snapshot=snapshot)
        if round_id is not None:
            return _redirect_round(round_id, extra=f"{redirect_flag}=1")
        return RedirectResponse(
            url=f"/?from_day={from_day_str}&to_day={to_day_str}&{redirect_flag}=1",
            status_code=303,
        )

    @app.post("/fill-lineup")
    async def fill_lineup(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        return _handle_lineup_action(form, redirect_flag="filled")

    @app.post("/apply-swap")
    async def apply_swap(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        round_id, key, date_from, date_to, from_day_str, to_day_str = _round_context_from_form(form)
        client = _tipsport_client()
        if round_id is not None:
            state, snapshot, _saved_at, _err = _resolve_round_by_id(round_id, client=client)
            read_only = False
        else:
            state, snapshot, read_only, _saved_at = _resolve_legacy_round(
                date_from=date_from,
                date_to=date_to,
                key=key,
                client=client,
            )
        if read_only:
            if round_id is not None:
                return _redirect_round(round_id)
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
            round_id=round_id,
        )
        if view.swap is not None:
            apply_swap_to_state(
                state,
                view.swap,
                remaining_match_ids=set(view.swap.remaining_match_ids),
            )
        _persist_round(key=key, state=state, snapshot=snapshot)
        if round_id is not None:
            return _redirect_round(round_id, extra="swapped=1")
        return RedirectResponse(
            url=f"/?from_day={from_day_str}&to_day={to_day_str}&swapped=1",
            status_code=303,
        )

    def _apply_calibrated_lineup(state: RoundGuiState, snapshot) -> bool:
        if state.calibration is None:
            return False
        lineup = build_lineup_for_knobs(
            tuple(snapshot.matches),
            state,
            knobs_from_snapshot(state.calibration),
        )
        if lineup is None:
            return False
        apply_lineup_to_state(state, lineup)
        return True

    @app.post("/calibrate-and-apply")
    async def calibrate_and_apply(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        round_id, key, date_from, date_to, from_day_str, to_day_str = _round_context_from_form(form)
        client = _tipsport_client()
        if round_id is not None:
            state, snapshot, _saved_at, _err = _resolve_round_by_id(round_id, client=client)
            read_only = False
        else:
            state, snapshot, read_only, _saved_at = _resolve_legacy_round(
                date_from=date_from,
                date_to=date_to,
                key=key,
                client=client,
            )
        if read_only:
            if round_id is not None:
                return _redirect_round(round_id)
            return RedirectResponse(
                url=f"/?from_day={from_day_str}&to_day={to_day_str}",
                status_code=303,
            )
        match_ids = [match.match_id for match in snapshot.matches]
        _apply_form_to_state(state, form=form, match_ids=match_ids)
        _persist_round(key=key, state=state, snapshot=snapshot)

        record = load_round_record(key)
        if record is None:
            raise RuntimeError(f"Round snapshot missing after persist: {key}")

        result = load_and_calibrate(
            key,
            sim_config=gui_calibration_config(state.field_size),
            quick=True,
        )
        state.calibration = calibration_snapshot_from_result(result)
        if not _apply_calibrated_lineup(state, snapshot):
            raise RuntimeError("Kalibrace proběhla, ale lineup se nepodařilo sestavit.")
        _persist_round(key=key, state=state, snapshot=snapshot)
        if round_id is not None:
            return _redirect_round(round_id, extra="calibrated=1")
        return RedirectResponse(
            url=f"/?from_day={from_day_str}&to_day={to_day_str}&calibrated=1",
            status_code=303,
        )

    @app.post("/apply-calibrated-lineup")
    async def apply_calibrated_lineup(request: Request) -> RedirectResponse:
        form = dict(await request.form())
        round_id, key, date_from, date_to, from_day_str, to_day_str = _round_context_from_form(form)
        client = _tipsport_client()
        if round_id is not None:
            state, snapshot, _saved_at, _err = _resolve_round_by_id(round_id, client=client)
            read_only = False
        else:
            state, snapshot, read_only, _saved_at = _resolve_legacy_round(
                date_from=date_from,
                date_to=date_to,
                key=key,
                client=client,
            )
        if read_only:
            if round_id is not None:
                return _redirect_round(round_id)
            return RedirectResponse(
                url=f"/?from_day={from_day_str}&to_day={to_day_str}",
                status_code=303,
            )
        match_ids = [match.match_id for match in snapshot.matches]
        _apply_form_to_state(state, form=form, match_ids=match_ids)
        if not _apply_calibrated_lineup(state, snapshot):
            if round_id is not None:
                return _redirect_round(round_id)
            return RedirectResponse(
                url=f"/?from_day={from_day_str}&to_day={to_day_str}",
                status_code=303,
            )
        _persist_round(key=key, state=state, snapshot=snapshot)
        if round_id is not None:
            return _redirect_round(round_id, extra="filled=1")
        return RedirectResponse(
            url=f"/?from_day={from_day_str}&to_day={to_day_str}&filled=1",
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
