# MegaX

Optimization engine for **Megatipovačka** (Tipsport/Chance free-to-play exact-score competition).

Given closing-line odds (1X2, U/O 2.5) and public money distribution, MegaX estimates score probabilities, models crowd behaviour, and selects lineups (8 matches + optional joker) to maximize the chance of winning a round — not just expected points.

Standalone repo under optagame (same layout as [deltax](https://github.com/maternaj/deltax), [sharpener](https://github.com/maternaj/sharpener)).

## Status

**v1 core complete** — Tipsport ingest, FastAPI GUI (:18555), P/C models, EV/GPP optimizer, Monte Carlo `simulate`, late-swap recommendations. See [`docs/design.md`](docs/design.md).

## CLI

```bash
# Fetch a Megatipovačka weekend window (comp 120, 1X2 + O/U 2.5, grouped by kickoff slot)
megax fetch-round --from-day 2026-07-25 --to-day 2026-07-27

# Poll FT results for match IDs (single shot or --watch until all finished)
megax poll-results --match-id 8212280
megax poll-results --match-id 8212280 --match-id 8212285 --watch

# Monte Carlo from saved round snapshot (see docs/simulate.md)
megax simulate --round 2026-07-24_2026-07-27 --universes 3000 --field 10000 --crowd-players 400
```

Session state is cached in `state/tipsport_scraper_state.json` (same init-web + cloudscraper pattern as deltax).

## GUI

```bash
# default http://0.0.0.0:18555
megax-gui
# or
./scripts/start_gui.sh
```

Skeleton panels: date window (Pá–Po default), Tipsport odds by kickoff slot, manual money % (Tipsport/Fortuna/Sazkabet), field size, rank/points A/B, tips + žolík, auto FT scoring from results API, **Optimizer** (A/B lineup fill), **Late swap** (protect/chase recommendations per remaining slot). Client-side auto-refresh toggle (default off, 90s). **P(x,y)** v3: 1X2 + team O/U + match totals blend + low-score fix → max-entropy; **C(x,y)** from money inputs.

Round snapshots (tips + last-known odds, no odds history) persist under `data/rounds/`. Past rounds open read-only from the saved snapshot.

## Competition context

| Topic | Detail |
|-------|--------|
| Format | 35 rounds × ~8 matches (Chance Liga), optional **joker** (point multiplier) |
| Primary prize | Large per-round payout + season leaderboard (125M CZK total pool) |
| Scoring | 10 / 6 / 4 / 2 / 0 (exact → winner+diff or goals → winner → goals only) |
| Accounts | 2 legal entries; tips editable until kickoff; early first-tip timestamp for tie-break |

## Modules

| Module | Status |
|--------|--------|
| `probability` | P(x,y) v3 — 1X2 + team O/U + match totals + Dixon–Coles |
| `scoring` | 10 / 6 / 4 / 2 / 0 |
| `ev` | EV(tip) = Σ P × points |
| `crowd` | C(x,y) from Fortuna soft money + shape (γ, δ, Prelec, β blend) |
| `utility` | GPP: U = EV / C^α |
| `lineup` | Two-account chalk/leverage + joker |
| `simulate` | Monte Carlo P(win) — CLI `megax simulate` — [`docs/simulate.md`](docs/simulate.md) |
| `swap` | Late swap protect/chase — GUI panel + apply |

**Next (v1.1):** partial refresh (results only, preserve form). **v2 (Linear backlog):** O/U money inputs, fan bias, Chance scrape, CS overlay.

## Setup

```bash
cd ~/optagame/megax
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
```

## Links

- [Megatipovačka rules (Chance)](https://www.chance.cz/souteze/detail/megatipovacka/3575/pravidla)
- GitHub: `https://github.com/maternaj/megax`
