"""HTML rendering for MegaX GUI."""

from __future__ import annotations

from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

from megax.crowd_observed import CROWD_GRID_SIZE, estimated_cells_from_crowd, prob_window
from megax.ev import expected_points, tip_points_distribution
from megax.gui.matrices import (
    build_ev_grid,
    build_utility_grid,
    render_editable_crowd_grid,
    render_matrix_table,
)
from megax.gui.jobs import job_store
from megax.optimize import estimate_optimize_seconds
from megax.gui.state import RoundGuiState, parse_tip_score
from megax.tipsport.results import MatchStatus

PRAGUE = ZoneInfo("Europe/Prague")
AUTO_REFRESH_DEFAULT_SECONDS = 90
AUTO_REFRESH_MIN_SECONDS = 30


def _fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return "—"
    return dt.astimezone(PRAGUE).strftime("%a %d.%m. %H:%M")


def _fmt_day(dt: datetime) -> str:
    return dt.astimezone(PRAGUE).date().isoformat()


def _status_label(result) -> str:
    if result is None:
        return "pending"
    if result.status == MatchStatus.FINISHED:
        score = f"{result.home_goals}:{result.away_goals}"
        return f"FT {score}"
    if result.status == MatchStatus.LIVE:
        return "live"
    return result.status.value


def _num_input(
    name: str,
    value: float | int | None,
    *,
    step: str = "0.1",
    width: str = "4.5em",
    placeholder: str = "",
    title: str = "",
    css_class: str = "",
) -> str:
    val = "" if value is None else str(value)
    placeholder_attr = f' placeholder="{escape(placeholder)}"' if placeholder else ""
    title_attr = f' title="{escape(title)}"' if title else ""
    class_attr = f' class="{escape(css_class)}"' if css_class else ""
    return (
        f'<input type="number" name="{escape(name)}" value="{escape(val)}" '
        f'step="{step}" min="0"{class_attr} style="width:{width};"{placeholder_attr}{title_attr}>'
    )


def _render_shell_error(
    *,
    message: str | None,
    error: str | None,
    round_id: int | None,
) -> str:
    rid = "" if round_id is None else str(round_id)
    msg_html = ""
    if message:
        msg_html += f'<p class="msg">{escape(message)}</p>'
    if error:
        msg_html += f'<p class="meta" style="color:var(--warn);">{escape(error)}</p>'
    return f"""<!DOCTYPE html>
<html lang="cs">
<head>
  <meta charset="utf-8">
  <title>MegaX — Megatipovačka</title>
  <style>
    body {{ font: 14px system-ui; background:#0f1419; color:#e7ecf1; padding:2rem; }}
    input, button {{ padding:.4rem .6rem; margin-right:.5rem; }}
    button {{ background:#4ea1ff; border:none; font-weight:600; cursor:pointer; }}
  </style>
</head>
<body>
  <h1>MegaX — Megatipovačka</h1>
  {msg_html}
  <form method="get" class="toolbar">
    <label for="round_id">roundId</label>
    <input id="round_id" type="number" name="round_id" value="{escape(rid)}" placeholder="383">
    <button type="submit">Načíst kolo</button>
  </form>
  <form method="post" action="/fetch-megatip" class="toolbar" style="margin-top:1rem;">
    <label for="fetch_round_id">roundId</label>
    <input id="fetch_round_id" type="number" name="round_id" value="{escape(rid)}" placeholder="383" required>
    <button type="submit">Obnovit top-3</button>
  </form>
  <p class="meta">Aktuální kolo lze auto-detekovat z veřejného Megatip API (bez loginu).</p>
</body>
</html>"""


def _round_hidden_fields(view: RoundView, date_from: str, date_to: str) -> str:
    if view.round_id is not None:
        parts = [f'<input type="hidden" name="round_id" value="{view.round_id}">']
        if view.state.round_number is not None:
            parts.append(
                f'<input type="hidden" name="round_number" value="{view.state.round_number}">'
            )
        return "".join(parts)
    return (
        f'<input type="hidden" name="from_day" value="{escape(date_from)}">'
        f'<input type="hidden" name="to_day" value="{escape(date_to)}">'
    )


def _megatip_refresh_button(round_id: int | None, read_only: bool) -> str:
    if read_only:
        return ""
    if round_id is not None:
        return (
            '<div class="toolbar" style="margin-top:.75rem;">'
            '<button type="submit" formaction="/fetch-megatip" class="secondary">'
            "Obnovit top-3</button>"
            "</div>"
        )
    return (
        '<div class="toolbar" style="margin-top:.75rem;">'
        '<label for="fetch_round_id">roundId</label> '
        '<input id="fetch_round_id" type="number" name="round_id" '
        'placeholder="383" style="width:6em;" required> '
        '<button type="submit" formaction="/fetch-megatip" class="secondary">'
        "Obnovit top-3</button>"
        "</div>"
    )


def _render_round_header(
    view: RoundView,
    round_id: int | None,
    round_label: str,
    date_from: str,
    date_to: str,
) -> str:
    if round_id is not None:
        selector = f"""
        <form method="get" class="toolbar">
          <div>
            <label for="round_id">roundId</label>
            <input id="round_id" type="number" name="round_id" value="{round_id}">
          </div>
          <div class="refresh-control">
            <label for="auto_refresh">
              <input id="auto_refresh" type="checkbox">
              Auto-refresh
            </label>
            <input id="auto_refresh_seconds" type="number" min="30" step="10" value="90" disabled>
            <span class="meta">s</span>
          </div>
          <button type="submit">Načíst kolo</button>
        </form>
        """
        meta = (
            f"{escape(round_label)} · {date_from} → {date_to} · "
            f"{len(view.snapshot.matches)} zápasů · "
            f'<span id="refresh-status">bez auto-refresh</span>'
        )
    else:
        selector = f"""
        <form method="get" class="toolbar">
          <div>
            <label for="from_day">Od</label>
            <input id="from_day" type="date" name="from_day" value="{escape(date_from)}">
          </div>
          <div>
            <label for="to_day">Do</label>
            <input id="to_day" type="date" name="to_day" value="{escape(date_to)}">
          </div>
          <div class="refresh-control">
            <label for="auto_refresh">
              <input id="auto_refresh" type="checkbox">
              Auto-refresh
            </label>
            <input id="auto_refresh_seconds" type="number" min="30" step="10" value="90" disabled>
            <span class="meta">s</span>
          </div>
          <button type="submit">Načíst kolo (legacy)</button>
        </form>
        """
        meta = (
            f"Legacy okno {escape(date_from)} → {escape(date_to)} · "
            f"{len(view.snapshot.matches)} zápasů"
        )

    return f"""
  <header>
    <div>
      <h1>MegaX — Megatipovačka</h1>
      <div class="meta">{meta}</div>
    </div>
    {selector}
  </header>
  """


def _render_megatip_panel(view: RoundView, *, round_id: int | None, read_only: bool) -> str:
    cache = view.state.megatip
    fetch_controls = _megatip_refresh_button(round_id, read_only)

    if cache is None:
        return f"""
        <section class="panel">
          <h3>Megatip (veřejné API)</h3>
          <p class="meta">Zatím bez dat — načtěte top-3 tipy davu pro každý zápas (bez loginu).</p>
          {fetch_controls}
        </section>
        """

    fetched = _fmt_dt_iso(cache.fetched_at) if cache.fetched_at else "—"
    missing = ""
    if cache.missing_match_ids:
        missing = (
            f'<p class="meta" style="color:var(--warn);">Tipsport chybí matchId: '
            f'{", ".join(str(x) for x in cache.missing_match_ids)}</p>'
        )
    round_num = cache.round_number or view.state.round_number
    round_label = f"kolo {round_num}" if round_num is not None else "—"
    return f"""
        <section class="panel">
          <h3>Megatip (veřejné API)</h3>
          <div class="summary">
            <div>{escape(round_label)}</div>
            <div>Top-3 načteno: <strong>{escape(fetched)}</strong></div>
          </div>
          {missing}
          {fetch_controls}
        </section>
        """


def _render_crowd_top3(state: RoundGuiState, match_id: int) -> str:
    cache = state.megatip
    if cache is None:
        return '<span class="meta">—</span>'
    match = cache.matches.get(str(match_id))
    if match is None or not match.top3:
        return '<span class="meta">—</span>'
    parts = []
    for label, pct in match.top3.items():
        mark = "* " if match.client_tip == label else ""
        parts.append(f"{mark}{escape(label)} {pct}%")
    client = ""
    if match.client_tip and match.client_tip not in match.top3:
        client = f'<div class="meta">my: {escape(match.client_tip)}</div>'
    return f'<div>{"".join(f"<div>{p}</div>" for p in parts)}</div>{client}'


def render_page(
    view: RoundView | None,
    *,
    message: str | None = None,
    error: str | None = None,
    round_id: int | None = None,
    poll_job: bool = False,
    job_expect: str | None = None,
) -> str:
    if view is None:
        return _render_shell_error(message=message, error=error, round_id=round_id)

    date_from = _fmt_day(view.snapshot.date_from)
    date_to = _fmt_day(view.snapshot.date_to)
    state = view.state
    account_a = state.accounts["A"]
    account_b = state.accounts["B"]
    round_id_val = view.round_id or state.round_id
    round_label = ""
    if round_id_val is not None:
        round_num = state.round_number or (state.megatip.round_number if state.megatip else None)
        round_label = f"roundId {round_id_val}"
        if round_num is not None:
            round_label = f"kolo {round_num} · {round_label}"
    read_only_note = ""
    if view.read_only:
        saved = _fmt_dt(view.saved_at) if view.saved_at else "—"
        read_only_note = (
            f'<p class="msg">Historické kolo — zobrazení uloženého snapshotu (kurzy + tipy k {saved}). '
            "C davu platí pro okamžik posledního uložení, ne pro čas tipování ostatních.</p>"
        )

    slot_blocks: list[str] = []
    for slot in view.slots:
        rows_html: list[str] = []
        for row in slot.matches:
            rows_html.append(_render_match_row(row, state))
        slot_blocks.append(
            f"""
            <section class="slot">
              <h2>Slot {_fmt_dt(slot.kickoff_at)}</h2>
              <table>
                <thead>
                  <tr>
                    <th>Zápas</th>
                    <th>Status</th>
                    <th>1X2</th>
                    <th>U/O 2.5</th>
                    <th>EV tip</th>
                    <th>EV/C tip</th>
                    <th>Tip A</th>
                    <th>Tip B</th>
                    <th>Body A</th>
                    <th>Body B</th>
                  </tr>
                </thead>
                <tbody>
                  {"".join(rows_html)}
                </tbody>
              </table>
            </section>
            """
        )

    joker_a_options = _joker_options(view, account_a.joker_match_id)
    joker_b_options = _joker_options(view, account_b.joker_match_id)

    message_html = ""
    if message:
        css = "msg" if not error else "meta"
        message_html = f'<p class="{css}">{escape(message)}</p>'
    if error:
        message_html += f'<p class="meta" style="color:var(--warn);">{escape(error)}</p>'
    message_html += read_only_note

    header_round = _render_round_header(view, round_id_val, round_label, date_from, date_to)
    megatip_panel = _render_megatip_panel(view, round_id=round_id_val, read_only=view.read_only)
    job_overlay = _job_progress_overlay(
        view,
        enabled=poll_job or job_store.is_running(view.round_key),
        expect_kind=job_expect,
    )

    return f"""<!DOCTYPE html>
<html lang="cs">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MegaX — Megatipovačka</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a222c;
      --text: #e7ecf1;
      --muted: #9aa7b5;
      --accent: #4ea1ff;
      --line: #2a3542;
      --ok: #57d38c;
      --warn: #f0b429;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.45 system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header, main, footer {{ padding: 1rem 1.25rem; }}
    header {{
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      align-items: center;
      justify-content: space-between;
    }}
    h1 {{ margin: 0; font-size: 1.25rem; }}
    h2 {{ margin: 0 0 .75rem; font-size: 1rem; color: var(--accent); }}
    .meta {{ color: var(--muted); font-size: .9rem; }}
    .panels {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1rem;
      margin-bottom: 1rem;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 1rem;
    }}
    .panel h3 {{ margin: 0 0 .75rem; font-size: .95rem; }}
    label {{ display: block; margin: .35rem 0 .15rem; color: var(--muted); font-size: .85rem; }}
    input, select, button {{
      background: #101820;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: .35rem .5rem;
    }}
    button {{
      background: var(--accent);
      color: #041018;
      border: none;
      font-weight: 600;
      cursor: pointer;
    }}
    button.secondary {{ background: #273141; color: var(--text); }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: .5rem; align-items: end; }}
    .refresh-control {{
      display: flex;
      flex-wrap: wrap;
      gap: .45rem;
      align-items: center;
      padding: .35rem .5rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #101820;
    }}
    .refresh-control label {{ display: inline-flex; align-items: center; gap: .35rem; margin: 0; cursor: pointer; }}
    .refresh-control input[type="number"] {{ width: 4.5em; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: .55rem .45rem;
      vertical-align: top;
      text-align: left;
    }}
    th {{ color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .03em; }}
    tr:last-child td {{ border-bottom: none; }}
    .fixture {{ font-weight: 600; }}
    .status-ft {{ color: var(--ok); }}
    .status-live {{ color: var(--warn); }}
    .crowd-edit input {{ background: #0d1218; border: 1px solid var(--line); color: var(--text); border-radius: 4px; padding: .15rem; font-size: .72rem; }}
    .crowd-cell-entered {{ background: rgba(78,161,255,0.18); }}
    .crowd-cell-computed {{ background: rgba(90,200,120,0.12); }}
    .crowd-cell-zero {{ background: rgba(120,120,120,0.12); }}
    .crowd-cell-empty {{ background: transparent; }}
    .crowd-computed-val {{ font-size: .68rem; color: #7fd99a; font-weight: 600; line-height: 1.1; }}
    .crowd-legend-entered {{ color: #6eb0ff; }}
    .crowd-legend-computed {{ color: #7fd99a; }}
    .mx-title-row {{ display: flex; justify-content: space-between; align-items: center; gap: .5rem; margin-bottom: .25rem; }}
    .mx-title-row .mx-title {{ margin-bottom: 0; }}
    .tip-rec strong {{ color: var(--accent); }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
    .summary strong {{ font-size: 1.1rem; }}
    .msg {{ color: var(--ok); }}
    footer {{ color: var(--muted); font-size: .85rem; border-top: 1px solid var(--line); }}
    .mx-row td {{ background: #121820; padding-top: 0; }}
    .mx-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }}
    .mx-wrap {{ border: 1px solid var(--line); border-radius: 8px; padding: .65rem; background: #101820; }}
    .mx-title {{ font-weight: 600; margin-bottom: .25rem; }}
    .mx-sub {{ color: var(--muted); font-size: .78rem; margin-bottom: .45rem; }}
    .mx-table {{ width: 100%; border-collapse: collapse; font-size: .72rem; }}
    .mx-table th, .mx-table td {{ border: 1px solid var(--line); text-align: center; padding: .2rem; }}
    .mx-table th {{ color: var(--muted); background: #0d1218; }}
    .mx-legend {{ color: var(--muted); font-size: .72rem; margin-top: .35rem; }}
    .mx-extra {{ color: var(--accent); font-size: .68rem; white-space: nowrap; }}
    .tip-dist {{ font-size: .72rem; line-height: 1.35; }}
  </style>
</head>
<body>
  {header_round}

  <main>
    {message_html}

    <form method="post" action="/save">
      {_round_hidden_fields(view, date_from, date_to)}

      <div class="panels">
        {megatip_panel}
        <section class="panel">
          <h3>Kolo</h3>
          <label for="field_size">Velikost pole (hráči)</label>
          {_num_input("field_size", state.field_size, step="1", width="8em")}
          <div class="summary" style="margin-top:1rem;">
            <div>FT vyhodnoceno: <strong>{view.finished_count}/{len(view.snapshot.matches)}</strong></div>
            <div>Kurzy: <strong>{_fmt_dt(view.fetched_at)}</strong></div>
            <div>Výsledky: <strong>{_fmt_dt(view.results_polled_at)}</strong></div>
          </div>
        </section>

        <section class="panel">
          <h3>Účet A</h3>
          <label>Pořadí</label>
          {_num_input("rank_a", account_a.rank, step="1", width="8em")}
          <label>Body (Chance)</label>
          {_num_input("points_a", account_a.points, step="1", width="8em")}
          <label>Průběh (auto)</label>
          <div><strong>{view.totals_a}</strong> bodů z tipů</div>
          <label for="joker_a">Žolík</label>
          <select id="joker_a" name="joker_a">{joker_a_options}</select>
        </section>

        <section class="panel">
          <h3>Účet B</h3>
          <label>Pořadí</label>
          {_num_input("rank_b", account_b.rank, step="1", width="8em")}
          <label>Body (Chance)</label>
          {_num_input("points_b", account_b.points, step="1", width="8em")}
          <label>Průběh (auto)</label>
          <div><strong>{view.totals_b}</strong> bodů z tipů</div>
          <label for="joker_b">Žolík</label>
          <select id="joker_b" name="joker_b">{joker_b_options}</select>
        </section>

        <section class="panel panel-strategy">
          <h3>Monte Carlo simulace</h3>
          {_render_simulation_panel(view, read_only=view.read_only)}
          <div class="toolbar" style="margin-top:.75rem;">
            <button type="submit">Uložit vstupy</button>
            <button type="submit" formaction="/refresh" class="secondary">Obnovit kurzy + výsledky</button>
          </div>
        </section>
      </div>

      {"".join(slot_blocks)}
    </form>
  </main>

  <footer>
    Display only — bez auto-submit na Chance. Neuložené tipy a matice C se při reloadu ztratí — nejdřív Uložit vstupy.
  </footer>
  {job_overlay}
  {_auto_refresh_script()}
</body>
</html>
"""


def _sim_agent_label(name: str) -> str:
    labels = {
        "saved_a": "Tip A",
        "saved_b": "Tip B",
        "pure_ev": "Pure EV",
        "pure_ev_joker": "EV + žolík",
        "ev_c": "Pure EV/C",
        "gpp": "Pure EV/C",
        "optimizer_a": "Optimizer A",
        "optimizer_b": "Optimizer B",
    }
    return labels.get(name, name)


def _render_simulation_panel(view: RoundView, *, read_only: bool) -> str:
    state = view.state
    total = len(view.snapshot.matches)
    with_p = sum(1 for slot in view.slots for row in slot.matches if row.probability is not None)
    with_c = sum(
        1
        for slot in view.slots
        for row in slot.matches
        if row.crowd is not None
        and (
            (row.crowd.known and any(any(r) for r in row.crowd.known))
            or (row.crowd.estimated and any(any(r) for r in row.crowd.estimated))
        )
    )
    parts = [
        '<p class="meta"><strong>Simulace</strong> — vyhodnotí aktuální tipy A/B vs. baselines (EV, EV/C). '
        "<strong>Optimalizace</strong> — MC hill-climb hledá lepší tipy A/B (top EV/EV/C kandidáti).</p>",
        f'<div class="summary">'
        f"<div>P model: <strong>{with_p}/{total}</strong></div>"
        f"<div>C matice: <strong>{with_c}/{total}</strong></div>"
        f"</div>",
    ]

    sim = state.last_simulation
    if sim is not None:
        if sim.error:
            parts.append(
                f'<p class="meta" style="color:var(--warn);">Simulace selhala: {escape(sim.error)}</p>'
            )
        elif sim.agents:
            if sim.skipped_match_ids:
                parts.append(
                    f'<p class="meta" style="color:var(--warn);">'
                    f"Ignorováno {len(sim.skipped_match_ids)} zápasů bez P matice "
                    f"(odložené / bez kurzů): {', '.join(str(x) for x in sim.skipped_match_ids)}"
                    f"</p>"
                )
            if sim.note:
                parts.append(f'<p class="meta">{escape(sim.note)}</p>')
            highlight = {"saved_a", "saved_b"}
            rows = []
            for agent in sim.agents:
                css = ' style="background:rgba(78,161,255,0.08);"' if agent.name in highlight else ""
                label = escape(_sim_agent_label(agent.name))
                rows.append(
                    f"<tr{css}>"
                    f"<td>{label}</td>"
                    f"<td>{agent.mean_points:.2f}</td>"
                    f"<td>{agent.p_win:.2%}</td>"
                    f"<td>{agent.p_top_10:.2%}</td>"
                    f"<td>{agent.p_top_100:.2%}</td>"
                    f"<td>{agent.p_top_1000:.2%}</td>"
                    f"</tr>"
                )
            parts.append(
                f'<p class="meta">Poslední běh: {_fmt_dt_iso(sim.simulated_at)} · '
                f"{sim.universes:,} universes · {sim.crowd_players:,} dav/universe · "
                f"pole {sim.field_size:,}</p>"
                '<table class="sim-table" style="width:100%;margin-top:.5rem;font-size:.82rem;">'
                "<thead><tr>"
                "<th>Strategie</th><th>Ø body</th><th>P(win)</th>"
                "<th>P top10</th><th>P top100</th><th>P top1k</th>"
                "</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table>"
            )

    parts.append(_render_optimizer_panel(view, read_only=read_only))

    if not read_only:
        default_uni = state.last_simulation.universes if state.last_simulation else 2000
        default_crowd = state.last_simulation.crowd_players if state.last_simulation else 400
        metric = state.last_optimization.metric if state.last_optimization else "top10"
        eval_est, sec_est = estimate_optimize_seconds(
            with_p,
            universes=default_uni,
        )
        time_est = _fmt_duration(sec_est)
        parts.append(
            '<div style="margin-top:1rem;padding-top:.75rem;border-top:1px solid var(--line);">'
            '<h4 style="margin:0 0 .5rem;font-size:.9rem;">Simulace (evaluator)</h4>'
            '<div class="refresh-control">'
            "<label>Universes "
            f'{_num_input("sim_universes", default_uni, step="500", width="6em")}'
            "</label>"
            "<label>Dav/universe "
            f'{_num_input("sim_crowd_players", default_crowd, step="50", width="5em")}'
            "</label>"
            '<button type="submit" formaction="/simulate">Spustit simulaci</button>'
            "</div>"
            f'<p class="meta">Vyhodnotí uložené tipy A/B. Odhad ~{_fmt_duration(default_uni * max(with_p, 1) * 0.0012)} '
            f"({default_uni:,} universes · {with_p} zápasů).</p>"
            "</div>"
            '<div style="margin-top:1rem;padding-top:.75rem;border-top:1px solid var(--line);">'
            '<h4 style="margin:0 0 .5rem;font-size:.9rem;">MC optimalizace tipů (QX-362)</h4>'
            "<label>Cíl "
            '<select name="opt_metric" style="width:8em;">'
            f'<option value="top10"{" selected" if metric == "top10" else ""}>P(top10)</option>'
            f'<option value="top1"{" selected" if metric == "top1" else ""}>P(top1)</option>'
            f'<option value="win"{" selected" if metric == "win" else ""}>P(win)</option>'
            "</select></label>"
            '<button type="submit" formaction="/optimize-lineup">Optimalizovat tipy (MC)</button>'
            "</div>"
            f'<p class="meta">Stejné universes/dav jako simulace výše. '
            f"Hill-climb ~{eval_est} kroků · odhad <strong>~{time_est}</strong> "
            f"(1× losování + rychlé přepočty).</p>"
        )
    elif sim is None:
        parts.append('<p class="meta">Simulace zatím neběžela.</p>')

    return "".join(parts)


def _opt_metric_label(metric: str) -> str:
    return {"top10": "P(top10)", "top1": "P(top1)", "win": "P(win)"}.get(metric, metric)


def _render_optimizer_panel(view: RoundView, *, read_only: bool) -> str:
    opt = view.state.last_optimization
    if opt is None:
        return ""

    parts = [
        '<div style="margin-top:1rem;padding-top:.75rem;border-top:1px solid var(--line);">',
        '<h4 style="margin:0 0 .5rem;font-size:.9rem;">Výsledek MC optimalizace</h4>',
    ]
    if opt.error:
        parts.append(
            f'<p class="meta" style="color:var(--warn);">Optimalizace selhala: {escape(opt.error)}</p>'
        )
    else:
        if opt.skipped_match_ids:
            parts.append(
                f'<p class="meta" style="color:var(--warn);">'
                f"Ignorováno {len(opt.skipped_match_ids)} zápasů bez P: "
                f"{', '.join(str(x) for x in opt.skipped_match_ids)}"
                f"</p>"
            )
        metric_label = _opt_metric_label(opt.metric)
        parts.append(
            f'<p class="meta">Poslední běh: {_fmt_dt_iso(opt.optimized_at)} · '
            f"cíl <strong>{metric_label}={opt.objective:.2%}</strong> · "
            f"{opt.search_evaluations} search evals · "
            f"{opt.universes:,} universes · {opt.crowd_players:,} dav/universe</p>"
        )
        if opt.note:
            parts.append(f'<p class="meta">{escape(opt.note)}</p>')
        parts.append(
            '<table class="sim-table" style="width:100%;margin-top:.5rem;font-size:.82rem;">'
            "<thead><tr>"
            "<th>Účet</th><th>Ø body</th><th>P(win)</th><th>P top10</th><th>P top100</th>"
            "</tr></thead><tbody>"
            f"<tr><td>MC opt A</td><td>{opt.mean_pts_a:.2f}</td>"
            f"<td>{opt.p_win_a:.2%}</td><td>{opt.p_top_10_a:.2%}</td>"
            f"<td>{opt.p_top_100_a:.2%}</td></tr>"
            f"<tr><td>MC opt B</td><td>{opt.mean_pts_b:.2f}</td>"
            f"<td>{opt.p_win_b:.2%}</td><td>{opt.p_top_10_b:.2%}</td>"
            f"<td>{opt.p_top_100_b:.2%}</td></tr>"
            "</tbody></table>"
        )
        if not read_only and opt.tips_a:
            parts.append(
                '<button type="submit" formaction="/apply-optimized-lineup" class="secondary" '
                'style="margin-top:.75rem;">Vyplnit Tip A/B z MC optimalizace</button>'
            )
    parts.append("</div>")
    return "".join(parts)


def _match_name(view: RoundView, match_id: int) -> str:
    for match in view.snapshot.matches:
        if match.match_id == match_id:
            return match.name
    return str(match_id)


def _render_account_lineup(view: RoundView, account) -> str:
    lineup = account
    picks = []
    for pick in lineup.picks:
        name = _match_name(view, pick.match_id)
        tag = "L" if pick.is_leverage else "C"
        picks.append(f"{escape(name)} {pick.tip} ({tag})")
    joker = _match_name(view, lineup.joker_match_id)
    body = (
        f"<div><strong>Účet {escape(lineup.account)}</strong> — "
        f"ΣEV {lineup.total_ev:.1f}, leverage {lineup.leverage_count}, žolík {escape(joker)}</div>"
        f'<div class="meta">{" · ".join(picks[:4])}</div>'
    )
    if len(picks) > 4:
        body += f'<div class="meta">{" · ".join(picks[4:])}</div>'
    return body


def _render_swap_panel(view: RoundView, *, read_only: bool) -> str:
    swap = view.swap
    if swap is None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        if view.snapshot.slots and all(slot.kickoff_at > now for slot in view.snapshot.slots):
            return (
                '<p class="meta">Late swap — aktivní až <strong>po startu prvního slotu</strong> '
                "(když už znáte průběžné body z Chance). Před tipováním použijte "
                "<strong>Kalibraci</strong> nebo Optimizer, ne late swap.</p>"
            )
        return '<p class="meta">Late swap — všechny sloty už začaly, nebo chybí P + C u zápasů.</p>'

    mode_labels = {
        "protect": "Protect (chalk)",
        "neutral": "Neutral",
        "chase": "Chase (YOLO)",
    }
    mode = mode_labels.get(swap.mode.value, swap.mode.value)
    next_slot = _fmt_dt(swap.next_slot_at) if swap.next_slot_at else "—"
    header = (
        f'<p class="meta">Režim: <strong>{escape(mode)}</strong> · '
        f"delta {swap.delta:.1f} bodů (lídr ~{swap.leader_estimate:.1f}) · "
        f"my {swap.our_best} (A {swap.our_points_a} / B {swap.our_points_b}) · "
        f"zbývá {swap.remaining_match_count} zápasů · další slot {escape(next_slot)}</p>"
    )

    if not swap.changes:
        return header + '<p class="meta">Aktuální tipy už sedí s doporučením pro zbývající sloty.</p>'

    rows = []
    for change in swap.changes:
        short = change.match_name.split(" - ")[0] if " - " in change.match_name else change.match_name
        rows.append(
            f"<tr><td>{escape(change.account)}</td><td>{escape(short)}</td>"
            f"<td>{escape(change.current_tip)} → <strong>{escape(change.recommended_tip)}</strong></td>"
            f"<td>{escape(change.pick_type)}</td></tr>"
        )
    table = (
        '<table style="width:100%;margin-top:.5rem;font-size:.85rem;">'
        "<thead><tr><th>Účet</th><th>Zápas</th><th>Tip</th><th>Typ</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    button = ""
    if not read_only:
        button = (
            '<button type="submit" formaction="/apply-swap" class="secondary" '
            'style="margin-top:.75rem;">Aplikovat swap tipy (zbývající sloty)</button>'
        )
    return header + table + button


def _render_calibration_panel(view: RoundView, *, read_only: bool) -> str:
    cal = view.state.calibration
    if cal is None and read_only:
        return ""

    parts = ['<div style="margin-top:1rem;padding-top:.75rem;border-top:1px solid var(--line);">']
    parts.append('<h4 style="margin:0 0 .5rem;font-size:.9rem;">Kalibrace (QX-329)</h4>')

    if cal is None:
        parts.append(
            '<p class="meta">Grid-search α / leverage / EV floor přes simulate (~4s). '
            "Vyplní tipy A/B optimálními knoby pro toto kolo.</p>"
        )
    else:
        chalk = (
            '<span style="color:var(--warn);"> · chalk mode</span>'
            if cal.use_chalk_mode
            else ""
        )
        parts.append(
            f'<p class="meta">Poslední: <strong>{escape(cal.label)}</strong>{chalk}<br>'
            f"P(win) A/B {cal.p_win_a:.2%} / {cal.p_win_b:.2%} · "
            f"joker {cal.p_win_pure_ev_joker:.2%} · "
            f"{cal.universes:,} universes · {cal.grid_size} platných kombinací<br>"
            f"α={cal.alpha_used:.3f} · {_fmt_dt_iso(cal.calibrated_at)}</p>"
        )

    if not read_only:
        parts.append(
            '<div style="display:flex;flex-wrap:wrap;gap:.5rem;margin-top:.5rem;">'
            '<button type="submit" formaction="/calibrate-and-apply" class="secondary">'
            "Kalibrovat + vyplnit tipy</button>"
        )
        if cal is not None:
            parts.append(
                '<button type="submit" formaction="/apply-calibrated-lineup" class="secondary">'
                "Znovu aplikovat uložené knoby</button>"
            )
        parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


def _fmt_dt_iso(iso_text: str) -> str:
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(iso_text.replace("Z", "+00:00"))
        return _fmt_dt(dt)
    except ValueError:
        return iso_text


def _render_lineup_panel(view: RoundView, *, read_only: bool) -> str:
    if view.lineup is None:
        return '<p class="meta">Lineup — potřeba P + C u všech zápasů (kurzy + dav %).</p>'

    lineup = view.lineup
    preview_note = ""
    if view.state.calibration is not None:
        preview_note = (
            f'<p class="meta">Náhled line-upu: kalibrované knoby ({escape(view.state.calibration.label)}).</p>'
        )
    parts = [
        preview_note or '<p class="meta">Mix chalk (EV) + leverage (GPP). Účty jsou vědomě nekorelované.</p>',
        _render_account_lineup(view, lineup.account_a),
        _render_account_lineup(view, lineup.account_b),
        f'<div class="meta">Leverage zápasy: {len(lineup.leverage_match_ids)}</div>',
    ]
    if not read_only:
        parts.append(
            '<button type="submit" formaction="/fill-lineup" class="secondary" '
            'style="margin-top:.75rem;">Vyplnit tipy A/B + žolíky</button>'
        )
    return "".join(parts)


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, int(round(seconds)))} s"
    minutes = int(round(seconds / 60))
    return f"{minutes} min"


def _job_progress_overlay(view: RoundView, *, enabled: bool, expect_kind: str | None = None) -> str:
    if not enabled:
        return ""
    round_id = view.round_id or view.state.round_id
    round_key = escape(view.round_key)
    from_day = escape(_fmt_day(view.snapshot.date_from))
    to_day = escape(_fmt_day(view.snapshot.date_to))
    round_id_js = "null" if round_id is None else str(int(round_id))
    expect_js = "null" if not expect_kind else f'"{escape(expect_kind)}"'
    return f"""<div id="job-overlay" style="
      position:fixed;inset:0;background:rgba(8,12,16,0.82);z-index:9999;
      display:flex;align-items:center;justify-content:center;padding:1rem;">
    <div style="background:var(--panel);border:1px solid var(--line);border-radius:12px;
      padding:1.25rem 1.5rem;max-width:28rem;width:100%;box-shadow:0 12px 40px rgba(0,0,0,.45);">
      <h3 style="margin:0 0 .75rem;font-size:1rem;" id="job-title">Monte Carlo běží…</h3>
      <div style="background:#0d1218;border-radius:999px;height:10px;overflow:hidden;margin-bottom:.75rem;">
        <div id="job-bar" style="height:100%;width:0%;background:var(--accent);transition:width .25s;"></div>
      </div>
      <p class="meta" id="job-detail" style="margin:0 0 .35rem;">Načítám stav…</p>
      <p class="meta" id="job-eta" style="margin:0;">Odhad zbývá: …</p>
    </div>
  </div>
<script>
(function () {{
  const roundKey = "{round_key}";
  const roundId = {round_id_js};
  const expectKind = {expect_js};
  const fromDay = "{from_day}";
  const toDay = "{to_day}";
  const bar = document.getElementById("job-bar");
  const detail = document.getElementById("job-detail");
  const eta = document.getElementById("job-eta");
  const title = document.getElementById("job-title");

  function fmtEta(sec) {{
    if (sec == null || !Number.isFinite(sec)) return "…";
    if (sec < 90) return Math.max(1, Math.round(sec)) + " s";
    return Math.round(sec / 60) + " min";
  }}

  function redirectDone(extra) {{
    if (roundId != null) {{
      window.location.href = "/?round_id=" + roundId + "&" + extra;
      return;
    }}
    window.location.href = "/?from_day=" + encodeURIComponent(fromDay)
      + "&to_day=" + encodeURIComponent(toDay) + "&" + extra;
  }}

  async function poll() {{
    try {{
      let statusUrl = "/api/job-status?round_key=" + encodeURIComponent(roundKey);
      if (expectKind) {{
        statusUrl += "&expect=" + encodeURIComponent(expectKind);
      }}
      const resp = await fetch(statusUrl);
      const data = await resp.json();
      if (!data.running && data.status === "idle") {{
        detail.textContent = "Běh nenalezen — obnovte stránku.";
        eta.textContent = "";
        return;
      }}
      if (data.status === "interrupted") {{
        title.textContent = "Běh přerušen";
        detail.textContent = data.message || "Spusťte znovu.";
        eta.textContent = "";
        bar.style.width = "100%";
        bar.style.background = "var(--warn)";
        return;
      }}
      const pct = data.percent != null ? data.percent : 0;
      bar.style.width = pct + "%";
      detail.textContent = data.message || "…";
      if (data.kind === "optimize") {{
        title.textContent = "MC optimalizace…";
      }} else if (data.kind === "simulate") {{
        title.textContent = "Monte Carlo simulace…";
      }}
      const elapsed = data.elapsed_seconds != null ? data.elapsed_seconds : 0;
      eta.textContent = "Uplynulo " + Math.round(elapsed) + " s · zbývá ~" + fmtEta(data.eta_seconds);
      if (data.running) {{
        setTimeout(poll, 600);
        return;
      }}
      if (data.status === "done") {{
        if (data.recovered) {{
          detail.textContent = data.message || "Hotovo — načítám výsledek…";
          bar.style.width = "100%";
        }}
        redirectDone(data.redirect_extra || "sim=1");
        return;
      }}
      if (data.status === "error") {{
        title.textContent = "Chyba";
        detail.textContent = data.error || data.message || "Neznámá chyba";
        eta.textContent = "";
        bar.style.width = "100%";
        bar.style.background = "var(--warn)";
      }}
    }} catch (err) {{
      detail.textContent = "Polling selhal — obnovte stránku.";
    }}
  }}
  poll();
}})();
</script>"""


def _auto_refresh_script() -> str:
    return f"""<script>
(function () {{
  const KEY_ENABLED = "megax_auto_refresh_enabled";
  const KEY_SECONDS = "megax_auto_refresh_seconds";
  const DEFAULT_SECONDS = {AUTO_REFRESH_DEFAULT_SECONDS};
  const MIN_SECONDS = {AUTO_REFRESH_MIN_SECONDS};

  const checkbox = document.getElementById("auto_refresh");
  const secondsInput = document.getElementById("auto_refresh_seconds");
  const status = document.getElementById("refresh-status");
  if (!checkbox || !secondsInput) {{
    return;
  }}

  function normalizeSeconds(raw) {{
    const parsed = parseInt(raw, 10);
    if (!Number.isFinite(parsed)) {{
      return DEFAULT_SECONDS;
    }}
    return Math.max(MIN_SECONDS, parsed);
  }}

  function applyRefresh() {{
    const enabled = checkbox.checked;
    const seconds = normalizeSeconds(secondsInput.value);
    secondsInput.value = String(seconds);
    secondsInput.disabled = !enabled;

    let meta = document.querySelector('meta[name="megax-auto-refresh"]');
    if (status) {{
      status.textContent = enabled ? `auto-refresh ${{seconds}}s` : "bez auto-refresh";
    }}
    if (!enabled) {{
      meta?.remove();
      return;
    }}
    if (!meta) {{
      meta = document.createElement("meta");
      meta.name = "megax-auto-refresh";
      meta.httpEquiv = "refresh";
      document.head.appendChild(meta);
    }}
    meta.content = String(seconds);
  }}

  function loadPrefs() {{
    checkbox.checked = localStorage.getItem(KEY_ENABLED) === "true";
    secondsInput.value = String(
      normalizeSeconds(localStorage.getItem(KEY_SECONDS) || DEFAULT_SECONDS)
    );
    applyRefresh();
  }}

  function savePrefs() {{
    localStorage.setItem(KEY_ENABLED, checkbox.checked ? "true" : "false");
    localStorage.setItem(KEY_SECONDS, String(normalizeSeconds(secondsInput.value)));
  }}

  checkbox.addEventListener("change", function () {{
    savePrefs();
    applyRefresh();
  }});
  secondsInput.addEventListener("change", function () {{
    savePrefs();
    applyRefresh();
  }});

  loadPrefs();
}})();
</script>"""


def _joker_options(view: RoundView, selected_id: int | None) -> str:
    options = ['<option value="">—</option>']
    for match in view.snapshot.matches:
        selected = " selected" if selected_id == match.match_id else ""
        options.append(f'<option value="{match.match_id}"{selected}>{escape(match.name)}</option>')
    return "".join(options)


def _fmt_tip_points_dist(prob, home: int, away: int) -> str:
    return tip_points_distribution(prob, home, away).fmt_compact()


def _render_ranked_tips(prob, candidates, *, value_text) -> str:
    if not candidates:
        return ""
    blocks: list[str] = []
    for index, candidate in enumerate(candidates[:3]):
        dist = _fmt_tip_points_dist(prob, candidate.home, candidate.away)
        val = value_text(candidate)
        if index == 0:
            blocks.append(
                f'<div class="tip-rec"><strong>{escape(candidate.score)}</strong> '
                f'<span class="meta">{escape(val)}</span></div>'
                f'<div class="meta tip-dist">{escape(dist)}</div>'
            )
        else:
            blocks.append(
                f'<div class="meta tip-dist">{escape(f"{candidate.score} {val} · {dist}")}</div>'
            )
    return "".join(blocks)


def _render_entered_tip_hint(prob, tip_text: str) -> str:
    tip = parse_tip_score(tip_text)
    if tip is None or prob is None:
        return ""
    home, away = tip
    ev = expected_points(prob, home, away)
    dist = _fmt_tip_points_dist(prob, home, away)
    return f'<div class="meta tip-dist">EV {ev:.2f} · {escape(dist)}</div>'


def _render_ev_cell(row: MatchRow) -> str:
    if row.analysis is None or row.probability is None:
        return '<span class="meta">—</span>'
    prob = row.probability
    return _render_ranked_tips(
        prob,
        row.analysis.ev.top,
        value_text=lambda c: f"{c.ev:.2f}",
    )


def _render_evc_cell(row: MatchRow) -> str:
    if row.analysis is None or row.probability is None:
        return '<span class="meta">—</span>'
    prob = row.probability
    return _render_ranked_tips(
        prob,
        row.analysis.gpp_top,
        value_text=lambda c: f"EV/C {c.utility:.1f}",
    )


def _render_match_row(row: MatchRow, state: RoundGuiState) -> str:
    match = row.match
    odds = match.odds
    result = row.result
    status = _status_label(result)
    status_class = "status-ft" if result and result.status == MatchStatus.FINISHED else (
        "status-live" if result and result.status == MatchStatus.LIVE else ""
    )
    match_key = str(match.match_id)
    tip_a = state.accounts["A"].tips.get(match_key, "")
    tip_b = state.accounts["B"].tips.get(match_key, "")

    over = "—" if odds.over_2_5 is None else f"{odds.over_2_5:.2f}"
    under = "—" if odds.under_2_5 is None else f"{odds.under_2_5:.2f}"
    ou = "—" if over == "—" or under == "—" else f"{under}u / {over}o"
    tip_a_hint = _render_entered_tip_hint(row.probability, tip_a)
    tip_b_hint = _render_entered_tip_hint(row.probability, tip_b)

    return f"""
    <tr>
      <td>
        <div class="fixture">{escape(match.name)}</div>
        <div class="meta">{_fmt_dt(match.kickoff_at)} · #{match.match_id}</div>
      </td>
      <td class="{status_class}">{escape(status)}</td>
      <td>{odds.home:.2f} / {odds.draw:.2f} / {odds.away:.2f}</td>
      <td>{ou}</td>
      <td>{_render_ev_cell(row)}</td>
      <td>{_render_evc_cell(row)}</td>
      <td><input name="tip_a_{match_key}" value="{escape(tip_a)}" placeholder="2:1" style="width:4em;">{tip_a_hint}</td>
      <td><input name="tip_b_{match_key}" value="{escape(tip_b)}" placeholder="1:1" style="width:4em;">{tip_b_hint}</td>
      <td>{"" if row.points_a is None else row.points_a}</td>
      <td>{"" if row.points_b is None else row.points_b}</td>
    </tr>
    {_render_match_matrices(row, state)}
    """


def _render_match_matrices(row: MatchRow, state: RoundGuiState) -> str:
    if row.probability is None:
        return (
            '<tr class="mx-row"><td colspan="10" class="meta">'
            "Matice P/C — chybí kurzy 1X2 nebo team O/U.</td></tr>"
        )
    prob = row.probability
    p_sub = (
        f"λ dom={prob.home_mu:.2f} λ host={prob.away_mu:.2f} · "
        f"P(1/X/2) {prob.p_home * 100:.0f}/{prob.p_draw * 100:.0f}/{prob.p_away * 100:.0f}"
    )
    p_html = render_matrix_table(
        prob.matrix,
        title="P(x,y) — model pravděpodobnosti",
        subtitle=p_sub,
        extra_values=build_ev_grid(prob.matrix),
        extra_label="EV",
        extra_decimals=2,
        legend_suffix="λ = očekávané góly · EV = očekávané body tipu (0–5 gólů)",
    )
    match_key = str(row.match.match_id)
    cells = state.crowd_cells_for_match(row.match.match_id)
    api_top3: dict[str, int] = {}
    if state.megatip is not None:
        mt = state.megatip.matches.get(match_key)
        if mt is not None:
            api_top3 = dict(mt.top3)

    estimated: dict[str, float] = {}
    if row.crowd is not None:
        estimated = estimated_cells_from_crowd(row.crowd)

    c_html = render_editable_crowd_grid(
        row.match.match_id,
        cells,
        grid_size=CROWD_GRID_SIZE,
        api_top3=api_top3,
        prob=prob_window(prob),
        estimated_cells=estimated,
        show_reset=bool(api_top3),
    )

    u_note = ""
    if row.analysis is not None and row.crowd is not None:
        filled = row.crowd.known
        if row.crowd.estimated is not None and filled is not None:
            filled = tuple(
                tuple(
                    row.crowd.known[i][j] or row.crowd.estimated[i][j]
                    for j in range(len(row.crowd.known[0]))
                )
                for i in range(len(row.crowd.known))
            )
        u_grid = build_utility_grid(
            prob_window(prob),
            row.crowd.matrix,
            alpha=row.analysis.gpp_alpha,
            known=filled,
        )
        u_note = (
            f'<div class="meta" style="margin-top:.5rem;">'
            f"EV/C = EV ÷ podíl davu (α=1) · nejlepší: "
            f"{escape(row.analysis.gpp_best.score)} EV/C {row.analysis.gpp_best.utility:.1f}"
            f"</div>"
        )
    else:
        u_grid = None

    if row.crowd is not None and row.crowd.note:
        c_html = c_html.replace(
            "</div>\n    ",
            f'<div class="mx-sub">{escape(row.crowd.note)}</div>\n    ',
            1,
        )

    return f"""
    <tr class="mx-row">
      <td colspan="10">
        <div class="mx-grid">{p_html}{c_html}</div>
        {u_note}
      </td>
    </tr>
    """
