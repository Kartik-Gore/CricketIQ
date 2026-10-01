"""Batting Feature Engineering Engine for CricketIQ."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.database.connection import engine
from sqlalchemy import text
from src.utils.metrics import (
    calculate_batting_average, calculate_strike_rate,
    calculate_dot_percentage, calculate_boundary_percentage,
    calculate_consistency_score
)

class BattingFeatureExtractor:
    """Computes advanced, context-aware batting performance features."""

    @staticmethod
    def extract_player_batting_features(player_name: str) -> Dict[str, Any]:
        """Extracts complete multidimensional batting features for a given player."""
        # 1. Base delivery query
        query = text("""
            SELECT 
                d.match_id, d.innings, d.over, d.batter_runs, d.extras, d.total_runs,
                d.is_legal, d.is_four, d.is_six, d.is_dot, d.phase, d.pressure_category,
                d.bowler_type, d.venue, m.team1, m.team2
            FROM deliveries d
            LEFT JOIN matches m ON d.match_id = m.match_id
            WHERE d.batter = :p
        """)
        deliveries = pd.read_sql(query, engine, params={"p": player_name})
        
        # Match stats for match-level aggregates
        pms_query = text("""
            SELECT match_id, bat_runs, balls_faced, is_out, date
            FROM player_match_stats
            WHERE player = :p AND balls_faced > 0
            ORDER BY date ASC
        """)
        match_stats = pd.read_sql(pms_query, engine, params={"p": player_name})

        if deliveries.empty or match_stats.empty:
            return {}

        total_runs = int(deliveries["batter_runs"].sum())
        balls_faced = int(deliveries["is_legal"].sum())
        fours = int(deliveries["is_four"].sum())
        sixes = int(deliveries["is_six"].sum())
        dots = int(deliveries["is_dot"].sum())
        total_innings = len(match_stats)
        dismissals = int(match_stats["is_out"].sum())

        singles = int((deliveries["batter_runs"] == 1).sum())
        twos = int((deliveries["batter_runs"] == 2).sum())

        # Consistency features
        scores = match_stats["bat_runs"].values
        runs_std, cv, consistency_score = calculate_consistency_score(scores)
        q25, q75 = np.percentile(scores, [25, 75]) if len(scores) >= 2 else (0.0, 0.0)
        iqr = round(float(q75 - q25), 2)
        median_runs = round(float(np.median(scores)), 1)

        # Phase breakdowns
        phase_groups = deliveries.groupby("phase")
        phase_stats = {}
        for phase in ["Powerplay", "Middle", "Death"]:
            if phase in phase_groups.groups:
                p_df = phase_groups.get_group(phase)
                p_runs = int(p_df["batter_runs"].sum())
                p_balls = int(p_df["is_legal"].sum())
                phase_stats[f"{phase.lower()}_runs"] = p_runs
                phase_stats[f"{phase.lower()}_balls"] = p_balls
                phase_stats[f"{phase.lower()}_strike_rate"] = calculate_strike_rate(p_runs, p_balls)
            else:
                phase_stats[f"{phase.lower()}_runs"] = 0
                phase_stats[f"{phase.lower()}_balls"] = 0
                phase_stats[f"{phase.lower()}_strike_rate"] = 0.0

        # Pace vs Spin splits
        bowler_type_groups = deliveries.groupby("bowler_type")
        pace_df = bowler_type_groups.get_group("Pace") if "Pace" in bowler_type_groups.groups else pd.DataFrame()
        spin_df = bowler_type_groups.get_group("Spin") if "Spin" in bowler_type_groups.groups else pd.DataFrame()

        pace_runs = int(pace_df["batter_runs"].sum()) if not pace_df.empty else 0
        pace_balls = int(pace_df["is_legal"].sum()) if not pace_df.empty else 0
        spin_runs = int(spin_df["batter_runs"].sum()) if not spin_df.empty else 0
        spin_balls = int(spin_df["is_legal"].sum()) if not spin_df.empty else 0

        # Innings Chase vs Defend
        chase_df = deliveries[deliveries["innings"] == 2]
        defend_df = deliveries[deliveries["innings"] == 1]
        chase_runs = int(chase_df["batter_runs"].sum())
        chase_balls = int(chase_df["is_legal"].sum())
        defend_runs = int(defend_df["batter_runs"].sum())
        defend_balls = int(defend_df["is_legal"].sum())

        # Pressure breakdown
        pressure_groups = deliveries.groupby("pressure_category")
        high_pressure_df = pressure_groups.get_group("High") if "High" in pressure_groups.groups else pd.DataFrame()
        hp_runs = int(high_pressure_df["batter_runs"].sum()) if not high_pressure_df.empty else 0
        hp_balls = int(high_pressure_df["is_legal"].sum()) if not high_pressure_df.empty else 0

        return {
            "player": player_name,
            "total_runs": total_runs,
            "balls_faced": balls_faced,
            "innings": total_innings,
            "dismissals": dismissals,
            "batting_average": calculate_batting_average(total_runs, dismissals),
            "strike_rate": calculate_strike_rate(total_runs, balls_faced),
            "fours": fours,
            "sixes": sixes,
            "boundary_count": fours + sixes,
            "boundary_pct": calculate_boundary_percentage((fours * 4) + (sixes * 6), total_runs),
            "dot_pct": calculate_dot_percentage(dots, balls_faced),
            "singles_pct": round((singles / max(1, balls_faced)) * 100.0, 2),
            "twos_pct": round((twos / max(1, balls_faced)) * 100.0, 2),
            "scoring_frequency": round(((balls_faced - dots) / max(1, balls_faced)) * 100.0, 2),
            "runs_per_innings": round(total_runs / max(1, total_innings), 2),
            "median_runs": median_runs,
            "runs_std": runs_std,
            "cv": cv,
            "iqr": iqr,
            "consistency_score": consistency_score,
            **phase_stats,
            "sr_vs_pace": calculate_strike_rate(pace_runs, pace_balls),
            "sr_vs_spin": calculate_strike_rate(spin_runs, spin_balls),
            "pace_balls": pace_balls,
            "spin_balls": spin_balls,
            "chase_strike_rate": calculate_strike_rate(chase_runs, chase_balls),
            "defend_strike_rate": calculate_strike_rate(defend_runs, defend_balls),
            "high_pressure_strike_rate": calculate_strike_rate(hp_runs, hp_balls),
            "high_pressure_runs": hp_runs
        }
