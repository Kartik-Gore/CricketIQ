"""Statistical formulas and metric calculations for cricket analytics."""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple

def calculate_batting_average(runs: float, dismissals: float) -> float:
    """Calculates batting average; returns runs if dismissals is 0."""
    if dismissals <= 0:
        return float(runs)
    return round(float(runs) / float(dismissals), 2)

def calculate_strike_rate(runs: float, balls_faced: float) -> float:
    """Calculates batting strike rate (runs per 100 balls)."""
    if balls_faced <= 0:
        return 0.0
    return round((float(runs) / float(balls_faced)) * 100.0, 2)

def calculate_bowling_economy(runs_conceded: float, legal_balls: float) -> float:
    """Calculates bowling economy rate (runs conceded per 6 legal balls)."""
    if legal_balls <= 0:
        return 0.0
    overs = legal_balls / 6.0
    return round(float(runs_conceded) / overs, 2)

def calculate_bowling_average(runs_conceded: float, wickets: float) -> float:
    """Calculates bowling average (runs per wicket)."""
    if wickets <= 0:
        return float("nan")
    return round(float(runs_conceded) / float(wickets), 2)

def calculate_bowling_strike_rate(legal_balls: float, wickets: float) -> float:
    """Calculates bowling strike rate (balls bowled per wicket)."""
    if wickets <= 0:
        return float("nan")
    return round(float(legal_balls) / float(wickets), 2)

def calculate_dot_percentage(dot_balls: float, total_balls: float) -> float:
    """Calculates percentage of dot balls."""
    if total_balls <= 0:
        return 0.0
    return round((float(dot_balls) / float(total_balls)) * 100.0, 2)

def calculate_boundary_percentage(boundary_runs: float, total_runs: float) -> float:
    """Calculates boundary percentage of runs scored."""
    if total_runs <= 0:
        return 0.0
    return round((float(boundary_runs) / float(total_runs)) * 100.0, 2)

def calculate_consistency_score(scores: Union[list, np.ndarray, pd.Series]) -> Tuple[float, float, float]:
    """
    Computes statistical consistency metrics:
    - Standard Deviation (std)
    - Coefficient of Variation (CV = std / mean)
    - Normalized Consistency Score (0-100 scale, where higher means more consistent)
    
    Formula:
    Normalized Consistency Score = max(0, min(100, 100 * (1 - min(CV, 1.5) / 1.5)))
    """
    arr = np.array(scores, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0, 0.0, 50.0 # Default midpoint for minimal samples
    
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if mean == 0:
        cv = 0.0
    else:
        cv = std / mean
        
    # Scale CV where typical T20 batter CV is 0.6 - 1.2
    # CV of 0.4 -> very consistent (~73 score)
    # CV of 1.4 -> high variance (~7 score)
    norm_score = max(0.0, min(100.0, 100.0 * (1.0 - (min(cv, 1.5) / 1.5))))
    return round(float(std), 2), round(float(cv), 2), round(norm_score, 1)

def calculate_exponential_weights(n_items: int, decay: float = 0.05) -> np.ndarray:
    """
    Computes normalized exponential weights for chronological items (most recent last).
    weight_i = exp(-decay * (n_items - 1 - i))
    """
    if n_items <= 0:
        return np.array([])
    ages = np.arange(n_items - 1, -1, -1)
    weights = np.exp(-decay * ages)
    return weights / np.sum(weights)

def calculate_pressure_index(
    innings: int,
    over: float,
    current_runs: int,
    wickets_lost: int,
    target_runs: Optional[int] = None
) -> float:
    """
    Calculates a normalized Match Pressure Index (0 to 100).
    For 2nd innings (chase):
      Based on Required Run Rate (RRR) vs baseline T20 par run rate (8.5),
      balls remaining, and wickets lost.
    For 1st innings:
      Based on match phase urgency and wickets lost relative to par.
    """
    balls_bowled = int(over) * 6 + int(round((over - int(over)) * 10))
    balls_remaining = max(1, 120 - balls_bowled)
    
    if innings == 2 and target_runs and target_runs > 0:
        runs_needed = max(0, target_runs - current_runs)
        rrr = (runs_needed / balls_remaining) * 6.0
        # Typical T20 baseline RRR is ~8.5
        rrr_diff = rrr - 8.5
        wickets_pressure = (wickets_lost / 10.0) * 35.0
        overs_factor = (120 - balls_remaining) / 120.0
        raw_pressure = 45.0 + (rrr_diff * 4.5) + wickets_pressure + (overs_factor * 10.0)
    else:
        # 1st innings pressure index
        # Par wickets at over x
        par_wickets = (balls_bowled / 120.0) * 6.0
        wicket_stress = max(0.0, (wickets_lost - par_wickets) * 8.0)
        phase_urgency = (balls_bowled / 120.0) * 30.0
        raw_pressure = 30.0 + wicket_stress + phase_urgency
        
    return round(float(np.clip(raw_pressure, 5.0, 99.0)), 1)
