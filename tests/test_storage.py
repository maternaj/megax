"""Tests for round JSON storage."""

from __future__ import annotations

from datetime import datetime, timezone

from megax.gui.state import RoundGuiState
from megax.storage import RoundRecord, load_round_record, save_round_record
from megax.team_mu import TeamOuLine
from megax.tipsport.offer import MatchOdds, MegaxMatch


def test_round_record_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("megax.storage.rounds_data_dir", lambda: tmp_path)
    match = MegaxMatch(
        match_id=1,
        name="A - B",
        home="A",
        away="B",
        kickoff_at=datetime(2026, 7, 25, 15, 0, tzinfo=timezone.utc),
        odds=MatchOdds(
            home=1.5,
            draw=4.0,
            away=6.0,
            over_2_5=1.8,
            under_2_5=2.0,
            home_team_lines=(TeamOuLine(0.5, 1.2, 3.0),),
            away_team_lines=(TeamOuLine(0.5, 1.7, 2.1),),
        ),
        match_type="PREMATCH",
        ended=False,
        competition_id=120,
    )
    state = RoundGuiState()
    state.accounts["A"].tips["1"] = "2:1"
    record = RoundRecord(
        round_key="2026-07-25_2026-07-27",
        state=state,
        matches=(match,),
        saved_at=datetime(2026, 7, 24, 8, 0, tzinfo=timezone.utc),
        fetched_at=datetime(2026, 7, 24, 8, 0, tzinfo=timezone.utc),
    )
    save_round_record(record)
    loaded = load_round_record("2026-07-25_2026-07-27")
    assert loaded is not None
    assert loaded.matches[0].odds.home_team_lines[0].line == 0.5
    assert loaded.state.accounts["A"].tips["1"] == "2:1"
