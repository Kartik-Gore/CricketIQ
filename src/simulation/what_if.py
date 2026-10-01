"""Scenario Simulation and What-If Counterfactual Modeling."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.ml.batting_model import BattingModelTrainer
from src.features.batting_features import BattingFeatureExtractor
from src.features.venue_features import VenueFeatureExtractor

class ScenarioSimulator:
    """Simulates performance outcomes under counterfactual match parameters."""

    @staticmethod
    def simulate_batting_scenario(
        player_name: str,
        venue: str,
        opposition: str,
        innings: int = 1,
        pressure_level: str = "Medium",
        batting_position: int = 3
    ) -> Dict[str, Any]:
        """
        Calculates scenario projection accounting for venue difficulty, pressure, and position.
        """
        # Baseline model prediction
        base_pred = BattingModelTrainer.predict_runs(player_name, innings=innings)
        expected_runs = base_pred["expected_runs"]

        # Adjust for venue difficulty
        venue_prof = VenueFeatureExtractor.get_venue_profile(venue)
        venue_diff = venue_prof.get("difficulty_index", 50.0)
        # 50 is neutral. If diff is 65 (bowler friendly), factor is ~0.90
        venue_multiplier = 1.0 + (50.0 - venue_diff) * 0.007

        # Adjust for pressure level
        pressure_multipliers = {
            "Low": 1.08,
            "Medium": 1.00,
            "High": 0.88
        }
        pressure_mult = pressure_multipliers.get(pressure_level, 1.0)

        # Adjust for batting position
        # Positions 1-3 have more balls available, 6-7 have fewer
        position_multipliers = {
            1: 1.15,
            2: 1.15,
            3: 1.05,
            4: 0.95,
            5: 0.80,
            6: 0.65,
            7: 0.50
        }
        pos_mult = position_multipliers.get(batting_position, 1.0)

        # Composite simulation calculation
        sim_runs = max(4.0, round(expected_runs * venue_multiplier * pressure_mult * pos_mult, 1))

        # Expected Strike Rate simulation
        bf = BattingFeatureExtractor.extract_player_batting_features(player_name)
        base_sr = bf.get("strike_rate", 130.0)
        # In death/high pressure or batting 5-7, strike rate goes up
        sr_boost = 1.10 if batting_position >= 5 else 1.0
        sim_sr = round(base_sr * sr_boost * (1.0 + (50.0 - venue_diff) * 0.003), 1)

        # Expected dismissal probability (typically 65% to 90% in T20 match)
        dismissal_prob = round(float(np.clip(0.70 + (venue_diff - 50.0) * 0.004, 0.40, 0.95)), 2)

        # Interval bounds
        margin = max(6.0, round(sim_runs * 0.38, 1))
        lower = max(0.0, round(sim_runs - margin, 1))
        upper = round(sim_runs + margin, 1)

        return {
            "player": player_name,
            "venue": venue,
            "opposition": opposition,
            "innings": innings,
            "pressure_level": pressure_level,
            "batting_position": batting_position,
            "expected_runs": sim_runs,
            "expected_strike_rate": sim_sr,
            "dismissal_probability": f"{dismissal_prob * 100:.0f}%",
            "lower_bound": lower,
            "upper_bound": upper,
            "range_str": f"{lower:.0f} - {upper:.0f} runs",
            "disclaimer": "Model-based counterfactual estimate under historical constraints."
        }

    @staticmethod
    def compare_two_scenarios(scenario_a: Dict[str, Any], scenario_b: Dict[str, Any]) -> pd.DataFrame:
        """Compares Scenario A vs Scenario B and calculates delta."""
        rows = [
            {"Metric": "Expected Runs", "Scenario A": scenario_a["expected_runs"], "Scenario B": scenario_b["expected_runs"]},
            {"Metric": "Expected Strike Rate", "Scenario A": scenario_a["expected_strike_rate"], "Scenario B": scenario_b["expected_strike_rate"]},
            {"Metric": "Dismissal Probability", "Scenario A": scenario_a["dismissal_probability"], "Scenario B": scenario_b["dismissal_probability"]},
            {"Metric": "Prediction Range", "Scenario A": scenario_a["range_str"], "Scenario B": scenario_b["range_str"]}
        ]
        df = pd.DataFrame(rows)
        # Calculate numerical difference where applicable
        diffs = []
        for r in rows:
            try:
                diff = float(r["Scenario B"]) - float(r["Scenario A"])
                diffs.append(f"{diff:+.1f}")
            except Exception:
                diffs.append("-")
        df["Difference (B - A)"] = diffs
        return df
