"""Tests for match total mu estimation."""

from __future__ import annotations

from megax.match_total_mu import (
    MatchTotalEstimate,
    blend_team_mus_with_match_total,
    estimate_match_total_mu,
)
from megax.team_mu import TeamMuEstimate, TeamOuLine


def test_estimate_match_total_mu_from_multiple_lines() -> None:
    lines = (
        TeamOuLine(2.5, 1.70, 2.19),
        TeamOuLine(3.0, 2.16, 1.71),
        TeamOuLine(2.0, 1.31, 3.43),
    )
    estimate = estimate_match_total_mu(lines)
    assert isinstance(estimate, MatchTotalEstimate)
    assert estimate.lines_used == 3
    assert 2.5 < estimate.total_mu < 3.5


def test_blend_preserves_home_away_ratio() -> None:
    team = TeamMuEstimate(home_mu=2.0, away_mu=1.0, home_lines_used=2, away_lines_used=1, source="x")
    total = MatchTotalEstimate(total_mu=2.4, lines_used=1, source="y")
    home_mu, away_mu = blend_team_mus_with_match_total(team, total, total_weight=1.0)
    assert abs(home_mu / away_mu - 2.0) < 1e-9
    assert abs(home_mu + away_mu - 2.4) < 1e-9
