"""Unit tests for analytics engines, matchups, and comparison."""

import pytest
from src.analytics.player_analysis import PlayerAnalyticsEngine
from src.analytics.matchup import MatchupEngine
from src.analytics.comparison import PlayerComparisonEngine
from src.analytics.venue import VenueAnalyticsEngine

def test_player_profile():
    p = PlayerAnalyticsEngine.get_complete_profile("V Kohli")
    assert p["name"] == "V Kohli"
    assert p["has_batting"] is True
    assert "batting_skill" in p
    assert len(p["batting_skill"]) == 10

def test_matchup_engine():
    matchup = MatchupEngine.get_matchup_detail("V Kohli", "JJ Bumrah")
    assert matchup["batter"] == "V Kohli"
    assert matchup["bowler"] == "JJ Bumrah"
    assert matchup["balls"] > 0
    assert matchup["strike_rate"] > 0

def test_player_comparison():
    comp = PlayerComparisonEngine.compare_players(["V Kohli", "DA Warner"])
    assert len(comp["profiles"]) == 2
    assert len(comp["summary_df"]) == 2
    assert "radar_data" in comp

def test_venue_summary():
    venues = VenueAnalyticsEngine.get_all_venues_summary()
    assert not venues.empty
    assert "difficulty_index" in venues.columns
