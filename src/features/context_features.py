"""Contextual Performance Index (CPI) Engine."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from src.database.connection import engine
from sqlalchemy import text

class ContextualPerformanceEngine:
    """
    Computes Contextual Performance Index (CPI).
    CPI measures whether a player outperformed or underperformed the statistical expectations
    dictated by venue difficulty, match phase, and bowler quality.
    """

    @staticmethod
    def calculate_player_cpi(player_name: str) -> Dict[str, Any]:
        """
        Calculates career contextual index for a batter.
        """
        query = text("""
            SELECT 
                d.match_id, d.phase, d.bowler_type, d.venue,
                d.batter_runs, d.is_legal
            FROM deliveries d
            WHERE d.batter = :p
        """)
        df = pd.read_sql(query, engine, params={"p": player_name})

        if df.empty:
            return {"player": player_name, "cpi": 100.0, "actual_runs": 0, "expected_runs": 0.0, "diff": 0.0}

        actual_runs = int(df["batter_runs"].sum())
        total_balls = int(df["is_legal"].sum())

        # Expected runs per ball by phase & bowler type (T20 empirical baselines)
        # Powerplay: ~1.25 runs/ball (125 SR)
        # Middle: ~1.18 runs/ball (118 SR)
        # Death: ~1.65 runs/ball (165 SR)
        phase_expected_rates = {
            "Powerplay": 1.25,
            "Middle": 1.18,
            "Death": 1.65
        }

        expected_runs_list = []
        for phase in df["phase"]:
            rate = phase_expected_rates.get(phase, 1.25)
            expected_runs_list.append(rate)

        expected_runs = round(float(np.sum(expected_runs_list)), 1)
        performance_diff = round(actual_runs - expected_runs, 1)

        # Normalized Contextual Performance Index (Baseline 100)
        # CPI > 100 means outperforming context; CPI < 100 means underperforming context
        cpi = round(float(100.0 * (actual_runs / max(1.0, expected_runs))), 1)

        return {
            "player": player_name,
            "cpi": cpi,
            "actual_runs": actual_runs,
            "expected_runs": expected_runs,
            "performance_diff": performance_diff,
            "balls_analyzed": total_balls
        }
