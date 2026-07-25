# Monte Carlo simulation (`megax simulate`)

Offline test of a **saved round snapshot** — compares lineup strategies under model **P(x,y)** (truth) and **C(x,y)** (crowd).

**Goal:** validate that your two-account mix beats pure chalk / pure GPP **before** submitting tips to Chance.  
**Not for:** live in-play decisions between slots → use **Late swap** in the GUI.

See also: [`design.md`](design.md) (architecture), [`README.md`](../README.md) (CLI overview).

---

## Prerequisites

1. Round loaded in GUI with **money %** filled for all matches (Fortuna minimum).
2. Tips and field size set → **Uložit vstupy**.
3. Snapshot exists at `data/rounds/{round_key}.json`  
   Example key: `2026-07-24_2026-07-27` (from–to dates in GUI).

---

## Command

```bash
cd ~/optagame/megax

# Quick check (~1 min)
megax simulate --round 2026-07-24_2026-07-27 \
  --universes 1000 --field 10000 --crowd-players 300 --seed 42

# More stable P(win) estimate (~5–10 min)
megax simulate --round 2026-07-24_2026-07-27 \
  --universes 5000 --field 10000 --crowd-players 500 --seed 42

# Save report to file
megax simulate --round 2026-07-24_2026-07-27 \
  --universes 3000 --field 10000 -o /tmp/sim-report.txt
```

Progress goes to **stderr** (so `-o` stays clean): setup line, then `Simulating: 2,500/5,000 (50%) — 12.3s, ~12s left`, then `Done in 24.5s`.

---

## Parameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--round` | *(required)* | Snapshot key matching `data/rounds/{key}.json` |
| `--universes` | 10 000 | How many independent “parallel worlds” to run. More → stabler P(win), slower. |
| `--field` | 50 000 | Conceptual field size. Sets default crowd sample: `min(field, 5000)` if `--crowd-players` omitted. |
| `--crowd-players` | `min(field, 5000)` | Virtual opponents sampled from **C** per universe. Not the full field — a random sample. |
| `--seed` | random | Fixed seed for reproducible comparisons (same tips, different strategies). |
| `--quiet` / `-q` | off | Suppress progress on stderr (report only). |

### Important nuances

- **GPP α and optimizer** use `field_size` from the **saved snapshot** (GUI “Velikost pole”), not necessarily `--field` on CLI. Change field size in GUI → save → re-run simulate.
- **`optimizer_*`** uses knobs from the last GUI **Kalibrace** when present (`ev`, `α×`, `lev` shown in the tips section). Without calibration, default leverage applies.
- **`--crowd-players`** is a **sample** of the crowd, not all 10 000 players. Lower values → **optimistic** P(win). Use 300–500 for speed, 2000–5000 for pessimistic estimates.
- P(win) from simulate is **not** your exact Chance win probability — it ranks strategies relative to each other.

---

## What each universe does

```
For each universe (1 … N):
  1. Sample one FT score per match from P(x,y)
  2. Generate crowd_players tip sheets from C(x,y)
  3. Score every crowd player + each agent (with joker ×2)
  4. Record: did agent beat the crowd? top 10? top 100?
```

---

## Agents in the report

| Agent | Description |
|-------|-------------|
| `pure_ev` | Best EV tip every match, no joker |
| `pure_ev_joker` | Same tips as `pure_ev`, joker on account A's match (fair EV baseline) |
| `gpp` | Best GPP (utility) tip every match |
| `optimizer_a` / `optimizer_b` | Lineup from `lineup.py` using **calibrated knobs** when the round has a calibration snapshot; otherwise default knobs |
| `saved_a` / `saved_b` | Tips stored in snapshot (your GUI inputs) |

If `saved_*` equals `optimizer_*`, your stored tips match the calibrated (or default) optimizer.  
If they differ, you edited tips after calibrate-and-apply or have not applied calibration yet.

The report ends with **Tips by agent** — kickoff order, joker match, and per-match scores for every agent.

---

## Output columns

| Column | Meaning |
|--------|---------|
| **Mean pts** | Average round points over all universes |
| **P(win)** | Share of universes where agent placed 1st vs simulated crowd (ties split) |
| **P top10 / top100 / top1k** | Share of universes where agent beat ≥90 / 99 / 99.9 % of crowd |

Example:

```
Agent           Mean pts   P(win)  P top10  P top100   P top1k
optimizer_a        28.64   3.12%   19.40%   67.30%  100.00%
pure_ev            25.16   0.55%    8.20%   52.50%  100.00%
```

→ Optimizer lineup wins ~6× more often than pure EV chalk in this model.

---

## When to run

| Timing | Use simulate? |
|--------|----------------|
| **Before round** — money filled, lineup ready, before Chance submit | **Yes** — main use case |
| **After big odds/money change** — re-save snapshot | Yes — compare before/after |
| **Between slots, live** | **No** — use Late swap panel + rank/body from Chance |
| **After slot 1 results** | Only if you re-save **new tips** for remaining matches and want a what-if |

---

## Workflow (recommended)

1. GUI: fill money → Optimizer → **Vyplnit tipy A/B** → tweak if needed → **Uložit vstupy**
2. `megax simulate --round … --universes 3000 --crowd-players 400 --seed 42`
3. Compare `saved_a/b` vs `pure_ev_joker` (apples-to-apples with joker) and vs each other
4. Optional: change field size in GUI (10k vs 50k), save, re-run — see if leverage mix improves P(win)
5. Submit tips to Chance manually

---

## Calibration (`megax calibrate`) — QX-329

Grid-search **GPP knobs** before submit. Uses fast simulate per combo.

```bash
# Quick grid (~27 combos, ~30s)
megax calibrate --round 2026-07-24_2026-07-27 --quick

# Full grid (~100 combos, ~2 min)
megax calibrate --round 2026-07-24_2026-07-27 --universes 1500 --crowd-players 400

# Custom slice
megax calibrate --round 2026-07-24_2026-07-27 \
  --ev-ratio 0.90,1.0 --alpha-mult 0.85,1.0 --leverage 0,1,2
```

**Searches:** `gpp_ev_ratio` (EV floor), `alpha_multiplier` (on field-size α), `leverage_count` (0 = all chalk).

**Output:** recommended knobs, lift vs current defaults, top-10 table, chalk-mode warning when `pure_ev_joker` beats optimizer.

**Note on O/U money (QX-322):** betting handle often skews Over; Megatipovačka crowd may favour 1:0 / 2:0 / 2:1 on favourites (mostly Under). Until O/U money is filled (~1h pre-kickoff), C tail may be off — calibration flags chalk mode when leverage loses to `pure_ev_joker`.

**GUI:** Optimizer panel → **Kalibrovat + vyplnit tipy** (~4s). Knobs + P(win) stats persist in the round snapshot; lineup preview uses calibrated settings. **Znovu aplikovat uložené knoby** re-fills A/B after money % updates without re-running simulate.

---

## Terminology (GUI columns)

| Term | Meaning |
|------|---------|
| **EV tip** | Score with highest expected points `EV = Σ P × points` |
| **GPP tip** | Score with highest utility `U = EV / C^α` (leverage vs crowd) |
| **Chalk** | High-crowd, safe tip (usually near EV tip) |
| **Leverage** | GPP-style pick on selected matches in the optimizer |
| **Late swap** | Mid-round tip changes for **remaining slots** only (GUI, not simulate) |

---

## Limitations (v1)

- Full-round tips only — no “given +12 pts after slot 1” conditional logic
- Crowd capped at 5000 players/universe for speed
- No tie-break timestamp rules from Chance
- Uses snapshot odds/money at save time — not live closing line unless you refresh + save first
- Vectorized engine (numpy): typical 2k universes × 400 crowd ≈ **1s** on a laptop (was ~2 min in v1 loop)

---

## Linear / issues

Implemented in [QX-321](https://linear.app/quantixx/issue/QX-321). CLI fix for circular import: commit `8c585d5`.
