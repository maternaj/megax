#!/usr/bin/env python3
"""Compare score models vs Tipsport correct-score odds for a Megatipovačka round."""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from megax.config import load_config
from megax.gui.weekend import default_round_window
from megax.ingest import fetch_round_snapshot
from megax.market_math import devig_two_way, p_over_poisson, poisson_pmf
from megax.probability import MODEL_VERSION, build_score_matrix_from_match, probability
from megax.score_prior import devig_1x2_probs, fit_market_score_prior
from megax.team_mu import TeamOuLine, estimate_team_mus
from megax.tipsport.client import TipsportClient
from megax.tipsport.offer import MegaxMatch, parse_match

EXACT_RESULT = "16-EXACT_RESULT-1"
ASIAN_TOTAL_FT = "16-ASIAN_TOTAL-1"
_LINE_RE = re.compile(r"(?:Více|Méně) než ([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class ScoreModel:
    name: str
    matrix: tuple[tuple[float, ...], ...]
    home_mu: float
    away_mu: float
    p_over_2_5: float | None


@dataclass(frozen=True)
class MatchAnalysis:
    match_id: int
    label: str
    cs_count: int
    cs_overround: float
    models: dict[str, ScoreModel]
    cs_fair: dict[str, float]
    cs_odds: dict[str, float]


def _line_from_name(name: str) -> float | None:
    match = _LINE_RE.search(name or "")
    return float(match.group(1)) if match else None


def _is_quarter_line(name: str) -> bool:
    return "(" in (name or "")


def parse_correct_scores(raw: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for event in raw.get("events") or []:
        if event.get("mySelectionId") != EXACT_RESULT:
            continue
        for opp in event.get("opps") or []:
            if not opp.get("bettingEnabled", True):
                continue
            score = str(opp.get("type") or opp.get("name") or "").strip()
            if re.fullmatch(r"\d+:\d+", score):
                scores[score] = float(opp["odd"])
    return scores


def power_devig(odds_by_score: dict[str, float]) -> dict[str, float]:
    odds = list(odds_by_score.values())
    lo, hi = 0.5, 1.5
    for _ in range(60):
        mid = (lo + hi) / 2.0
        total = sum((1.0 / odd) ** mid for odd in odds)
        if total > 1.0:
            lo = mid
        else:
            hi = mid
    exponent = (lo + hi) / 2.0
    raw = {score: (1.0 / odd) ** exponent for score, odd in odds_by_score.items()}
    total = sum(raw.values())
    return {score: value / total for score, value in raw.items()}


def parse_asian_total_lines(raw: dict[str, Any]) -> tuple[TeamOuLine, ...]:
    lines: dict[float, TeamOuLine] = {}
    for event in raw.get("events") or []:
        if event.get("mySelectionId") != ASIAN_TOTAL_FT:
            continue
        title = str(event.get("name") or "").lower()
        if "poločasu" in title:
            continue
        opps = [o for o in (event.get("opps") or []) if o.get("bettingEnabled", True)]
        if len(opps) != 2:
            continue
        if any(_is_quarter_line(str(o.get("name") or "")) for o in opps):
            continue
        line = _line_from_name(str(opps[0].get("name") or ""))
        if line is None:
            continue
        over = under = None
        for opp in opps:
            odd = float(opp["odd"])
            name = str(opp.get("name") or "")
            side = str(opp.get("type") or "").lower()
            if name.startswith("Více") or side == "o":
                over = odd
            elif name.startswith("Méně") or side == "u":
                under = odd
        if over is None or under is None:
            continue
        lines[line] = TeamOuLine(line=line, over=over, under=under)
    return tuple(lines[line] for line in sorted(lines))


def invert_total_mu(line: TeamOuLine, *, mu_max: float = 8.0) -> float | None:
    fair = devig_two_way(line.over, line.under)
    if fair is None:
        return None
    target = fair[0]
    lo, hi = 0.05, mu_max
    for _ in range(56):
        mid = (lo + hi) / 2.0
        p_mid = p_over_poisson(line.line, mid)
        if p_mid is None:
            return None
        if p_mid < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def estimate_match_total_mu(lines: tuple[TeamOuLine, ...]) -> float | None:
    estimates: list[float] = []
    for line in lines:
        mu = invert_total_mu(line)
        if mu is not None:
            estimates.append(mu)
    if not estimates:
        return None
    return sum(estimates) / len(estimates)


def build_totals_blended_model(
    match: MegaxMatch,
    raw: dict[str, Any],
    *,
    total_weight: float = 0.5,
    grid_size: int = 10,
) -> ScoreModel | None:
    """Blend team O/U lambdas with match Asian totals, then max-entropy fit to 1X2."""
    odds = match.odds
    probs_1x2 = devig_1x2_probs(odds.home, odds.draw, odds.away)
    team_estimate = estimate_team_mus(odds.home_team_lines, odds.away_team_lines)
    total_lines = parse_asian_total_lines(raw)
    total_mu = estimate_match_total_mu(total_lines)
    if probs_1x2 is None or team_estimate is None or total_mu is None:
        return None

    team_total = team_estimate.home_mu + team_estimate.away_mu
    if team_total <= 0:
        return None
    blended_total = (1.0 - total_weight) * team_total + total_weight * total_mu
    scale = blended_total / team_total
    home_mu = team_estimate.home_mu * scale
    away_mu = team_estimate.away_mu * scale

    prior = fit_market_score_prior(
        expected_total=home_mu + away_mu,
        goal_diff_away_home=away_mu - home_mu,
        p_home=probs_1x2[0],
        p_draw=probs_1x2[1],
        p_away=probs_1x2[2],
        grid_size=grid_size,
    )
    if prior is None:
        return None

    p_over_2_5 = None
    if odds.over_2_5 is not None and odds.under_2_5 is not None:
        fair_ou = devig_two_way(odds.over_2_5, odds.under_2_5)
        if fair_ou is not None:
            p_over_2_5 = fair_ou[0]

    return ScoreModel(
        name="v2_team_ou_plus_totals",
        matrix=prior.matrix,
        home_mu=home_mu,
        away_mu=away_mu,
        p_over_2_5=p_over_2_5,
    )


def build_cs_model(
    cs_fair: dict[str, float],
    *,
    grid_size: int = 10,
) -> ScoreModel:
    """Use de-vigged CS prices on listed cells; remaining grid mass stays zero."""
    grid = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    for score, prob in cs_fair.items():
        home, away = map(int, score.split(":"))
        if home < grid_size and away < grid_size:
            grid[home][away] = prob
    listed_mass = sum(sum(row) for row in grid)
    return ScoreModel(
        name="cs_power_devig",
        matrix=tuple(tuple(row) for row in grid),
        home_mu=float("nan"),
        away_mu=float("nan"),
        p_over_2_5=None,
    )


def _prob(model: ScoreModel, score: str) -> float:
    home, away = map(int, score.split(":"))
    grid = model.matrix
    if home >= len(grid) or away >= len(grid[0]):
        return 0.0
    return grid[home][away]


def _p_over_2_5(model: ScoreModel) -> float:
    total = 0.0
    for home, row in enumerate(model.matrix):
        for away, prob in enumerate(row):
            if home + away >= 3:
                total += prob
    return total


def _metrics(model: ScoreModel, cs_fair: dict[str, float]) -> dict[str, float]:
    diffs = [abs(_prob(model, score) - cs_fair[score]) for score in cs_fair]
    top_scores = sorted(cs_fair, key=cs_fair.get, reverse=True)[:10]
    top_diffs = [abs(_prob(model, score) - cs_fair[score]) for score in top_scores]
    kl = 0.0
    for score, target in cs_fair.items():
        pred = max(_prob(model, score), 1e-12)
        kl += target * math.log(target / pred)
    p10 = _prob(model, "1:0")
    p20 = _prob(model, "2:0")
    return {
        "mae_all": sum(diffs) / len(diffs),
        "mae_top10": sum(top_diffs) / len(top_diffs),
        "kl": kl,
        "p10_gap": p10 - cs_fair.get("1:0", 0.0),
        "p00_gap": _prob(model, "0:0") - cs_fair.get("0:0", 0.0),
        "order_10_20_wrong": 1.0 if p10 > p20 and cs_fair.get("2:0", 0.0) > cs_fair.get("1:0", 0.0) else 0.0,
        "matrix_over25": _p_over_2_5(model),
    }


def analyze_match(match: MegaxMatch, raw: dict[str, Any]) -> MatchAnalysis | None:
    cs_odds = parse_correct_scores(raw)
    if not cs_odds:
        return None
    cs_fair = power_devig(cs_odds)
    cs_overround = sum(1.0 / odd for odd in cs_odds.values())

    v3 = build_score_matrix_from_match(match)
    if v3 is None:
        return None
    v3_no_totals = build_score_matrix_from_match(match, total_blend_weight=0.0)
    v3_model = ScoreModel(
        name="v3",
        matrix=v3.matrix,
        home_mu=v3.home_mu,
        away_mu=v3.away_mu,
        p_over_2_5=v3.p_over_2_5,
    )
    no_totals_model = None
    if v3_no_totals is not None:
        no_totals_model = ScoreModel(
            name="v3_no_totals",
            matrix=v3_no_totals.matrix,
            home_mu=v3_no_totals.home_mu,
            away_mu=v3_no_totals.away_mu,
            p_over_2_5=v3_no_totals.p_over_2_5,
        )
    cs_model = build_cs_model(cs_fair)

    models = {v3_model.name: v3_model}
    if no_totals_model is not None:
        models[no_totals_model.name] = no_totals_model
    models[cs_model.name] = cs_model

    return MatchAnalysis(
        match_id=match.match_id,
        label=f"{match.home} vs {match.away}",
        cs_count=len(cs_odds),
        cs_overround=cs_overround,
        models=models,
        cs_fair=cs_fair,
        cs_odds=cs_odds,
    )


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_report(analyses: list[MatchAnalysis]) -> None:
    model_names = ["v3", "v3_no_totals", "cs_power_devig"]
    print("=" * 100)
    print("MEGAX ROUND ANALYSIS — model vs Tipsport correct score (power de-vig)")
    print("=" * 100)

    aggregate: dict[str, dict[str, list[float]]] = {
        name: {"mae_all": [], "mae_top10": [], "kl": [], "p10_gap": [], "p00_gap": []}
        for name in model_names
    }

    for item in analyses:
        print(f"\n{item.label} (#{item.match_id})")
        print(
            f"  CS outcomes: {item.cs_count} · raw overround: {(item.cs_overround - 1) * 100:.1f}%"
        )
        v3 = item.models["v3"]
        no_totals = item.models.get("v3_no_totals")
        print(
            f"  λ v3: {v3.home_mu:.2f} / {v3.away_mu:.2f} (total {v3.home_mu + v3.away_mu:.2f})"
        )
        if no_totals is not None:
            print(
                f"  λ team only: {no_totals.home_mu:.2f} / {no_totals.away_mu:.2f} "
                f"(total {no_totals.home_mu + no_totals.away_mu:.2f})"
            )
        market_over25 = None
        if v3.p_over_2_5 is not None:
            market_over25 = v3.p_over_2_5
            matrix_over25 = _p_over_2_5(v3)
            print(
                f"  P(over2.5): market {_fmt_pct(market_over25)} · v3 matrix {_fmt_pct(matrix_over25)}",
                end="",
            )
            if no_totals is not None:
                print(f" · team-only {_fmt_pct(_p_over_2_5(no_totals))}", end="")
            print()

        top = sorted(item.cs_fair, key=item.cs_fair.get, reverse=True)[:5]
        print("  Top CS scores:", ", ".join(f"{s} {_fmt_pct(item.cs_fair[s])}" for s in top))

        print(f"  {'model':<24} {'MAE all':>8} {'MAE top10':>10} {'KL':>7} {'Δ1:0':>7} {'Δ0:0':>7}")
        for name in model_names:
            model = item.models.get(name)
            if model is None:
                continue
            metrics = _metrics(model, item.cs_fair)
            for key in aggregate[name]:
                if key in metrics:
                    aggregate[name][key].append(metrics[key])
            print(
                f"  {name:<24}"
                f" {metrics['mae_all'] * 100:7.2f}%"
                f" {metrics['mae_top10'] * 100:9.2f}%"
                f" {metrics['kl']:7.3f}"
                f" {metrics['p10_gap'] * 100:+6.2f}%"
                f" {metrics['p00_gap'] * 100:+6.2f}%"
            )

        print("  Key scores (CS / v3 / team-only):")
        for score in ["1:0", "2:0", "2:1", "1:1", "0:0", "3:0"]:
            if score not in item.cs_fair:
                continue
            parts = [f"{score}", f"CS {_fmt_pct(item.cs_fair[score])}"]
            parts.append(f"v3 {_fmt_pct(_prob(v3, score))}")
            if no_totals is not None:
                parts.append(f"team {_fmt_pct(_prob(no_totals, score))}")
            print("    " + " · ".join(parts))

    print("\n" + "=" * 100)
    print("AGGREGATE (mean across matches)")
    print("=" * 100)
    print(f"{'model':<24} {'MAE all':>8} {'MAE top10':>10} {'KL':>7} {'Δ1:0':>7} {'Δ0:0':>7}")
    for name in model_names:
        bucket = aggregate[name]
        if not bucket["mae_all"]:
            continue
        print(
            f"{name:<24}"
            f" {sum(bucket['mae_all']) / len(bucket['mae_all']) * 100:7.2f}%"
            f" {sum(bucket['mae_top10']) / len(bucket['mae_top10']) * 100:9.2f}%"
            f" {sum(bucket['kl']) / len(bucket['kl']):7.3f}"
            f" {sum(bucket['p10_gap']) / len(bucket['p10_gap']) * 100:+6.2f}%"
            f" {sum(bucket['p00_gap']) / len(bucket['p00_gap']) * 100:+6.2f}%"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-day", help="Round lower bound YYYY-MM-DD")
    parser.add_argument("--to-day", help="Round upper bound YYYY-MM-DD")
    args = parser.parse_args(argv)

    if args.from_day and args.to_day:
        from megax.gui.weekend import day_bounds, parse_day

        date_from, _ = day_bounds(parse_day(args.from_day))
        _, date_to = day_bounds(parse_day(args.to_day))
    else:
        date_from, date_to = default_round_window()

    cfg = load_config()
    client = TipsportClient(cfg.tipsport_base_url, state_file=cfg.tipsport_state_file)
    snapshot = fetch_round_snapshot(date_from=date_from, date_to=date_to, config=cfg, client=client)

    bulk = client.fetch(f"/rest/external/offer/v1/matches?idCompetition={cfg.tipsport_competition_id}&allEvents=true")
    raw_by_id = {int(raw["id"]): raw for raw in (bulk.get("matches") or [])}

    analyses: list[MatchAnalysis] = []
    for match in snapshot.matches:
        raw = raw_by_id.get(match.match_id)
        if raw is None:
            print(f"skip {match.match_id}: missing raw offer", file=sys.stderr)
            continue
        parsed = parse_match(raw)
        if parsed is None:
            print(f"skip {match.match_id}: parse failed", file=sys.stderr)
            continue
        item = analyze_match(parsed, raw)
        if item is None:
            print(f"skip {match.match_id}: analysis failed", file=sys.stderr)
            continue
        analyses.append(item)

    print_report(analyses)
    print(f"\nMatches analyzed: {len(analyses)} · base model: {MODEL_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
