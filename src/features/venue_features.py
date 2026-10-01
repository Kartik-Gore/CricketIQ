"""Venue Intelligence and Stadium Index Feature Extraction."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.database.connection import engine
from sqlalchemy import text
from src.utils.metrics import calculate_strike_rate, calculate_bowling_economy

class VenueFeatureExtractor:
    """Extracts venue-level conditions, pace/spin bias, and individual player venue performance."""

    @staticmethod
    def get_venue_profile(venue_name: str) -> Dict[str, Any]:
        """Calculates comprehensive stadium metrics."""
        query = text("""
            SELECT 
                d.match_id, d.innings, d.batter_runs, d.total_runs, d.is_legal,
                d.is_bowler_wicket, d.is_four, d.is_six, d.is_dot, d.bowler_type
            FROM deliveries d
            WHERE d.venue = :v
        """)
        df = pd.read_sql(query, engine, params={"v": venue_name})

        if df.empty:
            return {"venue": venue_name, "total_matches": 0}

        total_matches = df["match_id"].nunique()
        total_runs = int(df["total_runs"].sum())
        total_legal = int(df["is_legal"].sum())
        total_wkts = int(df["is_bowler_wicket"].sum())
        boundaries = int((df["is_four"] | df["is_six"]).sum())

        avg_runs_per_match = round(total_runs / max(1, total_matches), 1)
        venue_economy = round((total_runs / max(1, total_legal)) * 6.0, 2)
        boundary_pct = round((boundaries / max(1, total_legal)) * 100.0, 2)

        # Innings splits (1st innings vs 2nd innings)
        inn1 = df[df["innings"] == 1]
        inn2 = df[df["innings"] == 2]

        inn1_matches = inn1["match_id"].nunique()
        inn1_runs = int(inn1["total_runs"].sum())
        avg_1st_innings = round(inn1_runs / max(1, inn1_matches), 1)

        inn2_matches = inn2["match_id"].nunique()
        inn2_runs = int(inn2["total_runs"].sum())
        avg_2nd_innings = round(inn2_runs / max(1, inn2_matches), 1)

        # Pace vs Spin splits at this venue
        pace_df = df[df["bowler_type"] == "Pace"]
        spin_df = df[df["bowler_type"] == "Spin"]

        pace_econ = calculate_bowling_economy(int(pace_df["total_runs"].sum()), int(pace_df["is_legal"].sum()))
        spin_econ = calculate_bowling_economy(int(spin_df["total_runs"].sum()), int(spin_df["is_legal"].sum()))

        pace_wkts = int(pace_df["is_bowler_wicket"].sum())
        spin_wkts = int(spin_df["is_bowler_wicket"].sum())

        # Venue Difficulty Index (0-100: higher means more difficult for batters / bowling friendly)
        # League average T20 economy is ~8.2
        league_avg_econ = 8.2
        diff_score = 50.0 + (league_avg_econ - venue_economy) * 12.0
        difficulty_index = round(float(np.clip(diff_score, 10.0, 95.0)), 1)

        return {
            "venue": venue_name,
            "total_matches": total_matches,
            "avg_match_runs": avg_runs_per_match,
            "avg_first_innings": avg_1st_innings,
            "avg_second_innings": avg_2nd_innings,
            "venue_economy": venue_economy,
            "boundary_pct": boundary_pct,
            "pace_economy": pace_econ,
            "spin_economy": spin_econ,
            "pace_wickets": pace_wkts,
            "spin_wickets": spin_wkts,
            "difficulty_index": difficulty_index
        }

    @staticmethod
    def get_player_venue_performance(player_name: str, venue_name: str) -> Dict[str, Any]:
        """Calculates specific player batting/bowling statistics at a target venue."""
        bat_query = text("""
            SELECT count(DISTINCT match_id) as matches, sum(batter_runs) as runs,
                   count(*) as balls, sum(is_four) as fours, sum(is_six) as sixes,
                   sum(CASE WHEN player_dismissed = :p THEN 1 ELSE 0 END) as dismissals
            FROM deliveries
            WHERE batter = :p AND venue = :v
        """)
        bat_df = pd.read_sql(bat_query, engine, params={"p": player_name, "v": venue_name})
        
        runs = int(bat_df["runs"].iloc[0]) if not bat_df.empty and pd.notna(bat_df["runs"].iloc[0]) else 0
        balls = int(bat_df["balls"].iloc[0]) if not bat_df.empty and pd.notna(bat_df["balls"].iloc[0]) else 0
        dismissals = int(bat_df["dismissals"].iloc[0]) if not bat_df.empty and pd.notna(bat_df["dismissals"].iloc[0]) else 0
        matches = int(bat_df["matches"].iloc[0]) if not bat_df.empty and pd.notna(bat_df["matches"].iloc[0]) else 0

        avg = round(runs / max(1, dismissals), 2) if dismissals > 0 else float(runs)
        sr = calculate_strike_rate(runs, balls)

        return {
            "player": player_name,
            "venue": venue_name,
            "matches": matches,
            "runs": runs,
            "balls": balls,
            "average": avg,
            "strike_rate": sr,
            "dismissals": dismissals
        }
