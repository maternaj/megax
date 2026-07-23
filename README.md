# MegaX

Optimization engine for **Megatipovačka** (Tipsport/Chance free-to-play exact-score competition).

Given closing-line odds (1X2, U/O 2.5) and public money distribution, MegaX estimates score probabilities, models crowd behaviour, and selects lineups (8 matches + optional joker) to maximize the chance of winning a round — not just expected points.

Standalone repo under optagame (same layout as [deltax](https://github.com/maternaj/deltax), [sharpener](https://github.com/maternaj/sharpener)).

## Status

**Brainstorm / design phase** — see [`docs/design.md`](docs/design.md) for problem framing, scoring rules, and planned architecture.

## Competition context

| Topic | Detail |
|-------|--------|
| Format | 35 rounds × ~8 matches (Chance Liga), optional **joker** (point multiplier) |
| Primary prize | Large per-round payout + season leaderboard (125M CZK total pool) |
| Scoring | 10 / 6 / 4 / 2 / 0 (exact → winner+diff or goals → winner → goals only) |
| Accounts | 2 legal entries; tips editable until kickoff; early first-tip timestamp for tie-break |

## Planned modules

| Module | Purpose |
|--------|---------|
| `probability` | P(x,y) from 1X2 + U/O closing odds (0:0–9:9) |
| `scoring` | Bodovací matice B(tip, actual) dle pravidel |
| `ev` | EV(tip) = Σ P(actual) × B(tip, actual) |
| `crowd` | C(x,y) from sharp vs soft 1X2 money + U/O bias |
| `utility` | GPP utility: EV vs crowd ownership, α(N) |
| `lineup` | 8-match portfolio + joker placement |
| `simulate` | Monte Carlo: truth × crowd × strategy agents |
| `swap` | Late swap state machine (live leaderboard → remaining picks) |

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
