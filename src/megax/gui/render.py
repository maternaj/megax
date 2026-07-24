"""HTML rendering for MegaX GUI."""

from __future__ import annotations

from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

from megax.gui.matrices import render_matrix_table
from megax.gui.service import MatchRow, RoundView
from megax.gui.state import BOOKMAKERS, MONEY_KEY_LABELS, MONEY_KEYS, RoundGuiState
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


def _money_inputs(match_id: int, state: RoundGuiState) -> str:
    parts: list[str] = []
    match_key = str(match_id)
    books = state.money.get(match_key, {})
    for book in BOOKMAKERS:
        values = books.get(book, {})
        cells = []
        for key in MONEY_KEYS:
            field = f"money_{match_key}_{book}_{key}"
            label = MONEY_KEY_LABELS[key]
            cells.append(
                _num_input(
                    field,
                    values.get(key),
                    step="0.1",
                    width="5em",
                    placeholder=label,
                    title=label,
                    css_class="money-input",
                )
            )
        parts.append(
            f'<div class="book"><span class="book-label">{escape(book)}</span>'
            f'<span class="money-grid">{"".join(cells)}</span></div>'
        )
    return "".join(parts)


def render_page(view: RoundView, *, message: str | None = None) -> str:
    date_from = _fmt_day(view.snapshot.date_from)
    date_to = _fmt_day(view.snapshot.date_to)
    state = view.state
    account_a = state.accounts["A"]
    account_b = state.accounts["B"]
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
                    <th>Peníze 1/X/2/U/O (Tips/Fort/Saz)</th>
                    <th>EV tip</th>
                    <th>GPP tip</th>
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
        message_html = f'<p class="msg">{escape(message)}</p>'
    message_html += read_only_note

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
    .book {{ margin-bottom: .35rem; }}
    .book-label {{ display: inline-block; width: 4.5rem; color: var(--muted); font-size: .78rem; text-transform: capitalize; }}
    .money-grid {{ display: inline-flex; gap: .25rem; flex-wrap: wrap; }}
    .money-grid input.money-input {{
      width: 5em;
      min-width: 5em;
      flex: 0 0 5em;
      padding-left: .3rem;
      padding-right: .3rem;
    }}
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
  </style>
</head>
<body>
  <header>
    <div>
      <h1>MegaX — Megatipovačka</h1>
      <div class="meta">Kolo {escape(date_from)} → {escape(date_to)} · {len(view.snapshot.matches)} zápasů · <span id="refresh-status">bez auto-refresh</span></div>
    </div>
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
      <button type="submit">Načíst kolo</button>
    </form>
  </header>

  <main>
    {message_html}

    <form method="post" action="/save">
      <input type="hidden" name="from_day" value="{escape(date_from)}">
      <input type="hidden" name="to_day" value="{escape(date_to)}">

      <div class="panels">
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

        <section class="panel">
          <h3>Optimizer</h3>
          {_render_lineup_panel(view, read_only=view.read_only)}
          <div class="toolbar">
            <button type="submit">Uložit vstupy</button>
            <button type="submit" formaction="/refresh" class="secondary">Obnovit kurzy + výsledky</button>
          </div>
        </section>

        <section class="panel">
          <h3>Late swap</h3>
          {_render_swap_panel(view, read_only=view.read_only)}
        </section>
      </div>

      {"".join(slot_blocks)}
    </form>
  </main>

  <footer>
    Display only — bez auto-submit na Chance. Neuložené tipy/peníze se při reloadu ztratí — nejdřív Uložit vstupy.
  </footer>
  {_auto_refresh_script()}
</body>
</html>
"""


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


def _render_lineup_panel(view: RoundView, *, read_only: bool) -> str:
    if view.lineup is None:
        return '<p class="meta">Lineup — potřeba P + C u všech zápasů (kurzy + peníze %).</p>'

    lineup = view.lineup
    parts = [
        '<p class="meta">Mix chalk (EV) + leverage (GPP). Účty jsou vědomě nekorelované.</p>',
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


def _fmt_tip_line(candidates, *, value_attr: str) -> str:
    parts = []
    for candidate in candidates[:3]:
        value = getattr(candidate, value_attr)
        parts.append(f"{candidate.score} {value:.2f}")
    return " · ".join(parts)


def _render_ev_cell(row: MatchRow) -> str:
    if row.analysis is None:
        return '<span class="meta">—</span>'
    best = row.analysis.ev.best
    top_line = escape(_fmt_tip_line(row.analysis.ev.top, value_attr="ev"))
    return (
        f'<div class="tip-rec"><strong>{escape(best.score)}</strong> '
        f'<span class="meta">{best.ev:.2f}</span></div>'
        f'<div class="meta">{top_line}</div>'
    )


def _render_gpp_cell(row: MatchRow) -> str:
    if row.analysis is None:
        return '<span class="meta">—</span>'
    best = row.analysis.gpp_best
    top_line = escape(_fmt_tip_line(row.analysis.gpp_top, value_attr="utility"))
    alpha = row.analysis.gpp_alpha
    return (
        f'<div class="tip-rec"><strong>{escape(best.score)}</strong> '
        f'<span class="meta">U {best.utility:.1f}</span></div>'
        f'<div class="meta">α={alpha:.2f} · {top_line}</div>'
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

    return f"""
    <tr>
      <td>
        <div class="fixture">{escape(match.name)}</div>
        <div class="meta">{_fmt_dt(match.kickoff_at)} · #{match.match_id}</div>
      </td>
      <td class="{status_class}">{escape(status)}</td>
      <td>{odds.home:.2f} / {odds.draw:.2f} / {odds.away:.2f}</td>
      <td>{ou}</td>
      <td>{_money_inputs(match.match_id, state)}</td>
      <td>{_render_ev_cell(row)}</td>
      <td>{_render_gpp_cell(row)}</td>
      <td><input name="tip_a_{match_key}" value="{escape(tip_a)}" placeholder="2:1" style="width:4em;"></td>
      <td><input name="tip_b_{match_key}" value="{escape(tip_b)}" placeholder="1:1" style="width:4em;"></td>
      <td>{"" if row.points_a is None else row.points_a}</td>
      <td>{"" if row.points_b is None else row.points_b}</td>
    </tr>
    {_render_match_matrices(row)}
    """


def _render_match_matrices(row: MatchRow) -> str:
    if row.probability is None:
        return (
            '<tr class="mx-row"><td colspan="11" class="meta">'
            "Matice P/C — chybí kurzy 1X2 nebo team O/U.</td></tr>"
        )
    prob = row.probability
    total_note = ""
    if prob.match_total_estimate is not None:
        total_note = (
            f" · totals {prob.match_total_estimate.lines_used}l"
            f" w={prob.total_blend_weight:.0%}"
        )
    p_sub = (
        f"λ dom={prob.home_mu:.2f} λ host={prob.away_mu:.2f} · "
        f"1X2 {prob.p_home * 100:.0f}/{prob.p_draw * 100:.0f}/{prob.p_away * 100:.0f} · "
        f"linek {prob.team_estimate.home_lines_used}+{prob.team_estimate.away_lines_used}"
        f"{total_note} · ρ={prob.low_score_rho:.2f}"
    )
    p_html = render_matrix_table(prob.matrix, title="P(x,y) — model pravděpodobnosti", subtitle=p_sub)
    if row.crowd is None:
        c_html = '<div class="mx-wrap"><div class="meta">C(x,y) — crowd model nedostupný</div></div>'
    else:
        crowd = row.crowd
        mass = crowd.outcome_mass
        raw = crowd.outcome_mass_raw
        raw_note = ""
        if raw is not None:
            raw_note = (
                f" (raw {raw[0] * 100:.0f}/{raw[1] * 100:.0f}/{raw[2] * 100:.0f})"
            )
        c_sub = (
            f"{crowd.note} · 1X2 dav "
            f"{mass[0] * 100:.0f}/{mass[1] * 100:.0f}/{mass[2] * 100:.0f}"
            f"{raw_note}"
        )
        c_html = render_matrix_table(
            crowd.matrix,
            title="C(x,y) — odhad tipů davu",
            subtitle=c_sub,
        )
    return f"""
    <tr class="mx-row">
      <td colspan="11">
        <div class="mx-grid">{p_html}{c_html}</div>
      </td>
    </tr>
    """
