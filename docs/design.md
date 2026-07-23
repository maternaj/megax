# MegaX — design notes

Analysis from initial brainstorm (2026-07). Czech source thread preserved in project history.

## Problem statement

Megatipovačka is a **large-field, free-to-play** exact-score competition. The objective is **not** to maximize expected points (EV) — that puts you in the top decile but sharing 1st place with hundreds of identical picks. The objective is to **maximize P(win the round)**, i.e. finish alone (or ahead on tie-breakers) at the top of a single 8-match slate.

We have:

- A model that derives **P(x,y)** (exact score probabilities) from closing 1X2 and U/O 2.5 odds (0:0–9:9).
- **Money distribution** on 1X2 from three sources (one sharp-friendly, two soft bookies).
- U/O money distribution — source TBD.
- **2 legal accounts** (not dozens).
- **Optional joker** per round (point multiplier on one match).
- **Late swap** — each match tip editable until kickoff; live round leaderboard available.

## Scoring rules (confirmed)

| Points | Condition |
|--------|-----------|
| **10** | Exact score |
| **6** | Correct winner/draw AND (correct goal difference OR correct **total** goals). Any non-exact draw tip when the match ends in a draw also scores 6. |
| **4** | Correct winner/draw only (wrong diff and wrong total goals) |
| **2** | Wrong winner/draw but correct **total** goals |
| **0** | Otherwise |

Joker multiplies points earned on that match.

Example (actual 4:1): 4:1→10; 3:0, 5:2→6; 5:1, 2:1→4; 2:3, 1:4→2; 0:2, 1:3→0.

## EV for a single tip

For tip T=(x,y) and actual S=(i,j):

```
EV(T) = Σ_{i,j} P(i,j) × B(T, S)
```

where B is the scoring function above. Compute over all candidate tips; pick argmax for pure-EV strategy.

**Pure EV is necessary but insufficient** for winning rounds — popular EV-max tips (2:0, 2:1, 1:0) cluster heavily in the field.

## Crowd model C(x,y)

Estimate fraction of players tipping each exact score.

### Preferred input (when available)

Ticket count on 1X2 and U/O — direct measure of recreational behaviour.

### Current input

Money % on 1X2 from **sharp** + **soft** bookies:

```
Delta(1) = M_soft(1) - M_sharp(1)   # "dumb money" on home win
```

Use sharp money as proxy for true probability; soft–sharp delta as public over/under bias.

### Decomposition

```
C(x,y) = M(outcome) × D(x,y | outcome)
```

- **M(outcome)** — money share on 1 / X / 2 (from soft bookies for Megatipovačka behaviour).
- **D(x,y | outcome)** — conditional score bias within that outcome (e.g. home win → 2:1 25%, 2:0 22%, 3:1 18%, …).

### U/O integration (planned)

Deform D dynamically: if crowd money is on Under 2.5, shift D toward 1:0, 2:0, 0:0; if Over, toward 2:1, 3:1, 3:2.

Without U/O money, use static D templates — weaker for defensive/tactical matchups.

## GPP utility (single match)

For field size N:

```
U(T) = EV(T) / C(T)^α     or     U(T) = EV(T) - α(N) × C(T)
```

Higher α for larger N → more contrarian picks. α tuned via Monte Carlo.

**Joker placement:**

```
JokerValue(match) = EV(best_tip) / C(best_tip)
```

Place joker where leverage is highest, not necessarily on the biggest favourite.

## Round-level optimization (8 matches)

Each round is a **short slate** — high variance, large per-round prize.

### Chalk & contrarian mix

- **5–6 anchors (chalk):** high-EV, likely crowd picks — base points.
- **2–3 leverage picks:** decent EV, low C(x,y) — separation if they hit.

Avoid galaxy-brain 8× contrarian — too many zeros.

### Two-account strategy

| Account | Role |
|---------|------|
| **A — Aggressive** | Joker on Friday early kickoff → immediate information. Hit → switch remaining picks to chalk (protect lead). Miss → chaos mode on weekend. |
| **B — Optimistic** | Joker on Sunday highest-EV/leverage match. Covers scenario where weekend is volatile and nobody accumulated points. |

Tickets should be **negatively correlated** — not two variants of the same chalk sheet.

### Late swap state machine

After each kickoff block:

1. Read live round points + leaderboard position (confirmed available).
2. Compute **delta** to estimated leader score.
3. **Small delta** → chalk remaining picks (block chasers).
4. **Large delta** → raise α, contrarian / YOLO on remaining matches.

## Monte Carlo evaluation

1. **Truth:** sample season/round outcomes from P(x,y) per match (10k universes).
2. **Crowd:** generate N virtual players tipping from C(x,y).
3. **Agents:** test strategies (pure EV, mild GPP, aggressive GPP, two-account portfolios).
4. **Metric:** P(1st place in round), not mean points.

Use to calibrate α, chalk/contrarian ratio, joker policy.

## Operational playbook

1. **Day 1 of competition:** submit any dummy tip immediately (tie-break timestamp).
2. **Before each match:** refresh closing odds → P matrix → optimize → swap tip.
3. **Joker:** decide per account strategy (early vs optimal-EV slot).

## Known weaknesses / open questions

| # | Risk | Mitigation |
|---|------|------------|
| 1 | **Betting money ≠ free-play picks** — recreational Megatipovačka may be more emotional (3:1 for favourite team) | Sharp/soft delta still useful; may need Megatipovačka-specific calibration later |
| 2 | **Poisson tail / game state** — 2:0 more "sticky" than model suggests | Manual context adjustments; don't trust 3:2 leverage blindly |
| 3 | **Fan bias** — local patriotism not in bookie money | Derby/home-team correction factor on C |
| 4 | **Fanouškovská tipovačka conflict** — same tips for team vs solo leaderboard | Choose primary objective (solo round win) |
| 5 | **Tie-breaker ambiguity** — "fewer submitted matches" may mean skip low-EV games | Verify with Chance support; defer in v1 |
| 6 | **No U/O money yet** | Static D templates until source found |
| 7 | **Exact-score noise** — 93rd-minute goal turns 10 pts into 4 | Accept variance; 8-match slate is inherently lottery-heavy |
| 8 | **Single-entry limit** (2 accounts) — can't cover all scenarios | Negative correlation + MC validation |

## Implementation phases (proposed)

1. **Core math** — scoring matrix, EV calculator, unit tests against rule examples.
2. **Crowd model** — C(x,y) from 1X2 money + static D; plug-in for U/O later.
3. **Single-round optimizer** — 8 picks + joker, utility function, two negatively correlated lineups.
4. **Simulator** — MC backtest on historical odds if available.
5. **Late swap** — state machine wired to live leaderboard input.
6. **Data pipeline** — odds ingestion, closing-line scheduler, export format for manual tip entry.

## Data schema (sketch)

```yaml
round:
  id: "2026-kolo-12"
  matches:
    - id: "sparta-teplice"
      kickoff: "2026-08-15T18:00+02:00"
      odds:
        home: 1.45
        draw: 4.50
        away: 7.00
        over_2_5: 1.85
        under_2_5: 1.95
      money:
        sharp:  { home: 0.62, draw: 0.18, away: 0.20 }
        soft_1: { home: 0.85, draw: 0.05, away: 0.10 }
        soft_2: { home: 0.82, draw: 0.06, away: 0.12 }
      joker_eligible: true
```

Model output per match: `P[10][10]`, `C[10][10]`, recommended tip, EV, U-score.
