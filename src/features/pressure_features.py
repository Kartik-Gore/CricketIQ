"""Pressure modeling and high-leverage situational analytics."""

import pandas as pd
from typing import Dict, Any
from src.database.connection import engine
from sqlalchemy import text
from src.utils.metrics import calculate_strike_rate, calculate_bowling_economy

class PressureFeatureExtractor:
    """Computes player performance metrics across statistical match-pressure levels."""

    @staticmethod
    def get_player_pressure_profile(player_name: str) -> Dict[str, Any]:
        """Calculates player batting and bowling splits under Low, Medium, High pressure."""
        # Batting query
        bat_query = text("""
            SELECT pressure_category, count(*) as balls, sum(batter_runs) as runs,
                   sum(is_four) as fours, sum(is_six) as sixes,
                   sum(CASE WHEN player_dismissed = :p THEN 1 ELSE 0 END) as dismissals
            FROM deliveries
            WHERE batter = :p
            GROUP BY pressure_category
        """)
        bat_df = pd.read_sql(bat_query, engine, params={"p": player_name})

        bat_splits = {}
        for cat in ["Low", "Medium", "High"]:
            row = bat_df[bat_df["pressure_category"] == cat]
            if not row.empty:
                balls = int(row["balls"].iloc[0])
                runs = int(row["runs"].iloc[0])
                dism = int(row["dismissals"].iloc[0])
                bat_splits[f"bat_{cat.lower()}_balls"] = balls
                bat_splits[f"bat_{cat.lower()}_runs"] = runs
                bat_splits[f"bat_{cat.lower()}_sr"] = calculate_strike_rate(runs, balls)
                bat_splits[f"bat_{cat.lower()}_avg"] = round(runs / max(1, dism), 2)
            else:
                bat_splits[f"bat_{cat.lower()}_balls"] = 0
                bat_splits[f"bat_{cat.lower()}_runs"] = 0
                bat_splits[f"bat_{cat.lower()}_sr"] = 0.0
                bat_splits[f"bat_{cat.lower()}_avg"] = 0.0

        # Bowling query
        bowl_query = text("""
            SELECT pressure_category, count(*) as balls, sum(total_runs) as runs,
                   sum(is_bowler_wicket) as wickets
            FROM deliveries
            WHERE bowler = :p
            GROUP BY pressure_category
        """)
        bowl_df = pd.read_sql(bowl_query, engine, params={"p": player_name})

        bowl_splits = {}
        for cat in ["Low", "Medium", "High"]:
            row = bowl_df[bowl_df["pressure_category"] == cat]
            if not row.empty:
                balls = int(row["balls"].iloc[0])
                runs = int(row["runs"].iloc[0])
                wkts = int(row["wickets"].iloc[0])
                bowl_splits[f"bowl_{cat.lower()}_balls"] = balls
                bowl_splits[f"bowl_{cat.lower()}_econ"] = calculate_bowling_economy(runs, balls)
                bowl_splits[f"bowl_{cat.lower()}_wkts"] = wkts
            else:
                bowl_splits[f"bowl_{cat.lower()}_balls"] = 0
                bowl_splits[f"bowl_{cat.lower()}_econ"] = 0.0
                bowl_splits[f"bowl_{cat.lower()}_wkts"] = 0

        return {
            "player": player_name,
            **bat_splits,
            **bowl_splits
        }
