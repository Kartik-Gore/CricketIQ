"""Unit tests for feature engineering extractors."""

import pytest
from src.features.batting_features import BattingFeatureExtractor
from src.features.bowling_features import BowlingFeatureExtractor
from src.features.form_features import FormFeatureExtractor
from src.features.context_features import ContextualPerformanceEngine

def test_batting_features_existing_player():
    bf = BattingFeatureExtractor.extract_player_batting_features("V Kohli")
    assert bf["total_runs"] > 0
    assert bf["balls_faced"] > 0
    assert "powerplay_strike_rate" in bf
    assert "middle_strike_rate" in bf
    assert "death_strike_rate" in bf
    assert 0 <= bf["consistency_score"] <= 100

def test_batting_features_nonexistent_player():
    # Edge case: nonexistent player
    bf = BattingFeatureExtractor.extract_player_batting_features("Unknown NonExistent")
    assert bf == {}

def test_bowling_features_existing_player():
    bw = BowlingFeatureExtractor.extract_player_bowling_features("JJ Bumrah")
    assert bw["wickets"] > 0
    assert bw["economy"] > 0
    assert "powerplay_economy" in bw

def test_form_feature_extractor():
    form = FormFeatureExtractor.calculate_player_form("V Kohli", role="batting", window=5)
    assert form["recent_form_score"] > 0
    assert form["trend"] in ["Improving", "Stable", "Declining", "Volatile"]

def test_contextual_performance_index():
    cpi = ContextualPerformanceEngine.calculate_player_cpi("V Kohli")
    assert cpi["cpi"] > 0
    assert cpi["actual_runs"] > 0
