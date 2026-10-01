"""Bowling Feature Engineering Engine for CricketIQ."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from src.database.connection import engine
from sqlalchemy import text
from src.utils.metrics import (
    calculate_bowling_economy, calculate_bowling_average,
    calculate_bowling_strike_rate, calculate_dot_percentage,
    calculate_consistency_score
)

class BowlingFeatureExtractor:
    """Computes multidimensional bowling performance metrics and situational splits."""

    @staticmethod
    def extract_player_bowling_features(player_name: str) -> Dict[str, Any]:
        """Extracts complete bowling profile for a given bowler."""
        query = text("""
            SELECT 
                d.match_id, d.innings, d.over, d.total_runs, d.is_legal,
                d.is_dot, d.is_bowler_wicket, d.is_four, d.is_six,
                d.phase, d.pressure_category, d.batter_hand
            FROM deliveries d
            WHERE d.bowler = :p
        """)
        deliveries = pd.read_sql(query, engine, params={"p": player_name})

        pms_query = text("""
            SELECT match_id, balls_bowled, runs_conceded, wickets, date
            FROM player_match_stats
            WHERE player = :p AND balls_bowled > 0
            ORDER BY date ASC
        """)
        match_stats = pd.read_sql(pms_query, engine, params={"p": player_name})

        if deliveries.empty or match_stats.empty:
            return {}

        total_deliveries = len(deliveries)
        legal_balls = int(deliveries["is_legal"].sum())
        runs_conceded = int(deliveries["total_runs"].sum())
        wickets = int(deliveries["is_bowler_wicket"].sum())
        dots = int(deliveries["is_dot"].sum())
        boundaries = int((deliveries["is_four"] | deliveries["is_six"]).sum())

        overs = round(legal_balls / 6.0, 1)
        economy = calculate_bowling_economy(runs_conceded, legal_balls)
        bowling_avg = calculate_bowling_average(runs_conceded, wickets)
        strike_rate = calculate_bowling_strike_rate(legal_balls, wickets)
        dot_pct = calculate_dot_percentage(dots, legal_balls)
        boundary_pct = round((boundaries / max(1, legal_balls)) * 100.0, 2)

        # Hauls
        four_wkt_hauls = int((match_stats["wickets"] == 4).sum())
        five_wkt_hauls = int((match_stats["wickets"] >= 5).sum())

        # Consistency of wickets and runs conceded per match
        wkt_scores = match_stats["wickets"].values
        _, _, wkt_consistency = calculate_consistency_score(wkt_scores)

        # Phase breakdowns
        phase_groups = deliveries.groupby("phase")
        phase_stats = {}
        for phase in ["Powerplay", "Middle", "Death"]:
            if phase in phase_groups.groups:
                p_df = phase_groups.get_group(phase)
                p_legal = int(p_df["is_legal"].sum())
                p_runs = int(p_df["total_runs"].sum())
                p_wkts = int(p_df["is_bowler_wicket"].sum())
                phase_stats[f"{phase.lower()}_overs"] = round(p_legal / 6.0, 1)
                phase_stats[f"{phase.lower()}_economy"] = calculate_bowling_economy(p_runs, p_legal)
                phase_stats[f"{phase.lower()}_wickets"] = p_wkts
            else:
                phase_stats[f"{phase.lower()}_overs"] = 0.0
                phase_stats[f"{phase.lower()}_economy"] = 0.0
                phase_stats[f"{phase.lower()}_wickets"] = 0

        # Batter Hand matchups (RHB vs LHB)
        hand_groups = deliveries.groupby("batter_hand")
        rhb_df = hand_groups.get_group("Right-hand bat") if "Right-hand bat" in hand_groups.groups else pd.DataFrame()
        lhb_df = hand_groups.get_group("Left-hand bat") if "Left-hand bat" in hand_groups.groups else pd.DataFrame()

        rhb_legal = int(rhb_df["is_legal"].sum()) if not rhb_df.empty else 0
        rhb_runs = int(rhb_df["total_runs"].sum()) if not rhb_df.empty else 0
        rhb_wkts = int(rhb_df["is_bowler_wicket"].sum()) if not rhb_df.empty else 0

        lhb_legal = int(lhb_df["is_legal"].sum()) if not lhb_df.empty else 0
        lhb_runs = int(lhb_df["total_runs"].sum()) if not lhb_df.empty else 0
        lhb_wkts = int(lhb_df["is_bowler_wicket"].sum()) if not lhb_df.empty else 0

        return {
            "player": player_name,
            "matches": len(match_stats),
            "overs": overs,
            "legal_balls": legal_balls,
            "runs_conceded": runs_conceded,
            "wickets": wickets,
            "economy": economy,
            "bowling_average": bowling_avg if pd.notna(bowling_avg) else 0.0,
            "strike_rate": strike_rate if pd.notna(strike_rate) else 0.0,
            "dot_pct": dot_pct,
            "boundary_pct": boundary_pct,
            "four_wkt_hauls": four_wkt_hauls,
            "five_wkt_hauls": five_wkt_hauls,
            "wkt_consistency_score": wkt_consistency,
            **phase_stats,
            "economy_vs_rhb": calculate_bowling_economy(rhb_runs, rhb_legal),
            "wickets_vs_rhb": rhb_wkts,
            "economy_vs_lhb": calculate_bowling_economy(lhb_runs, lhb_legal),
            "wickets_vs_lhb": lhb_wkts
        }
