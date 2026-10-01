"""Unit tests for metric formulas, consistency scores, and edge cases."""

import pytest
import numpy as np
from src.utils.metrics import (
    calculate_batting_average, calculate_strike_rate,
    calculate_bowling_economy, calculate_bowling_average,
    calculate_bowling_strike_rate, calculate_dot_percentage,
    calculate_boundary_percentage, calculate_consistency_score,
    calculate_pressure_index
)

def test_batting_average_standard():
    assert calculate_batting_average(100, 2) == 50.0

def test_batting_average_zero_dismissals():
    # Edge case: Not out batter
    assert calculate_batting_average(75, 0) == 75.0

def test_strike_rate():
    assert calculate_strike_rate(50, 25) == 200.0
    # Edge case: zero balls faced
    assert calculate_strike_rate(0, 0) == 0.0

def test_bowling_economy():
    # 24 balls = 4.0 overs, 32 runs conceded -> 8.00 econ
    assert calculate_bowling_economy(32, 24) == 8.0
    # Zero balls bowled
    assert calculate_bowling_economy(10, 0) == 0.0

def test_bowling_average_and_strike_rate():
    # 60 runs for 3 wickets -> 20.0 avg
    assert calculate_bowling_average(60, 3) == 20.0
    # Zero wickets -> NaN
    assert np.isnan(calculate_bowling_average(30, 0))
    # Strike rate: 24 balls, 2 wickets -> 12 balls/wkt
    assert calculate_bowling_strike_rate(24, 2) == 12.0

def test_consistency_score():
    # Consistent scores
    scores_consistent = [45, 48, 50, 47, 52]
    std, cv, norm_score = calculate_consistency_score(scores_consistent)
    assert cv < 0.15
    assert norm_score > 80.0

    # Highly volatile scores
    scores_volatile = [0, 85, 0, 92, 4]
    std, cv, norm_score = calculate_consistency_score(scores_volatile)
    assert cv > 1.0
    assert norm_score < 50.0

def test_pressure_index_bounds():
    # Test chase pressure
    p_high = calculate_pressure_index(innings=2, over=18.0, current_runs=150, wickets_lost=7, target_runs=185)
    assert p_high >= 70.0
    # Lower bound
    p_low = calculate_pressure_index(innings=1, over=2.0, current_runs=18, wickets_lost=0)
    assert p_low <= 45.0
