"""Data transformation, feature derivation, and metadata enrichment."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.utils.logger import logger
from src.utils.helpers import get_match_phase
from src.utils.metrics import calculate_pressure_index

# Curated reference database of top IPL/international players' styles and roles
PLAYER_PROFILES = {
    # Batters & All-rounders
    "V Kohli": {"hand": "Right-hand bat", "style": "Right-arm medium", "role": "Batter"},
    "RG Sharma": {"hand": "Right-hand bat", "style": "Right-arm offbreak", "role": "Batter"},
    "MS Dhoni": {"hand": "Right-hand bat", "style": "Right-arm medium", "role": "Wicketkeeper-Batter"},
    "AB de Villiers": {"hand": "Right-hand bat", "style": "Right-arm medium", "role": "Batter"},
    "DA Warner": {"hand": "Left-hand bat", "style": "Right-arm legbreak", "role": "Batter"},
    "KL Rahul": {"hand": "Right-hand bat", "style": "None", "role": "Wicketkeeper-Batter"},
    "SK Raina": {"hand": "Left-hand bat", "style": "Right-arm offbreak", "role": "Batter"},
    "S Dhawan": {"hand": "Left-hand bat", "style": "Right-arm offbreak", "role": "Batter"},
    "CH Gayle": {"hand": "Left-hand bat", "style": "Right-arm offbreak", "role": "Batter"},
    "AD Russell": {"hand": "Right-hand bat", "style": "Right-arm fast", "role": "All-rounder"},
    "HH Pandya": {"hand": "Right-hand bat", "style": "Right-arm fast-medium", "role": "All-rounder"},
    "RA Jadeja": {"hand": "Left-hand bat", "style": "Slow left-arm orthodox", "role": "All-rounder"},
    "GJ Maxwell": {"hand": "Right-hand bat", "style": "Right-arm offbreak", "role": "All-rounder"},
    "SA Yadav": {"hand": "Right-hand bat", "style": "Right-arm offbreak", "role": "Batter"},
    "RR Pant": {"hand": "Left-hand bat", "style": "None", "role": "Wicketkeeper-Batter"},
    "SV Samson": {"hand": "Right-hand bat", "style": "None", "role": "Wicketkeeper-Batter"},
    "Shubman Gill": {"hand": "Right-hand bat", "style": "Right-arm offbreak", "role": "Batter"},
    "YBK Jaiswal": {"hand": "Left-hand bat", "style": "Right-arm legbreak", "role": "Batter"},
    "F du Plessis": {"hand": "Right-hand bat", "style": "Right-arm legbreak", "role": "Batter"},
    "Q de Kock": {"hand": "Left-hand bat", "style": "None", "role": "Wicketkeeper-Batter"},
    "JC Buttler": {"hand": "Right-hand bat", "style": "None", "role": "Wicketkeeper-Batter"},
    "KA Pollard": {"hand": "Right-hand bat", "style": "Right-arm medium", "role": "All-rounder"},
    "SR Watson": {"hand": "Right-hand bat", "style": "Right-arm fast-medium", "role": "All-rounder"},
    "DJ Bravo": {"hand": "Right-hand bat", "style": "Right-arm medium-fast", "role": "All-rounder"},
    "SP Narine": {"hand": "Left-hand bat", "style": "Right-arm offbreak", "role": "Bowling All-rounder"},
    "AR Patel": {"hand": "Left-hand bat", "style": "Slow left-arm orthodox", "role": "All-rounder"},
    "MP Stoinis": {"hand": "Right-hand bat", "style": "Right-arm medium", "role": "All-rounder"},
    "N Pooran": {"hand": "Left-hand bat", "style": "Right-arm offbreak", "role": "Wicketkeeper-Batter"},
    "H Klaasen": {"hand": "Right-hand bat", "style": "Right-arm offbreak", "role": "Wicketkeeper-Batter"},
    "RK Singh": {"hand": "Left-hand bat", "style": "Right-arm offbreak", "role": "Batter"},
    "RD Gaikwad": {"hand": "Right-hand bat", "style": "Right-arm offbreak", "role": "Batter"},
    "TM Head": {"hand": "Left-hand bat", "style": "Right-arm offbreak", "role": "Batter"},

    # Specialist Bowlers
    "JJ Bumrah": {"hand": "Right-hand bat", "style": "Right-arm fast", "role": "Bowler"},
    "YS Chahal": {"hand": "Right-hand bat", "style": "Right-arm legbreak", "role": "Bowler"},
    "Rashid Khan": {"hand": "Right-hand bat", "style": "Right-arm legbreak", "role": "Bowling All-rounder"},
    "R Ashwin": {"hand": "Right-hand bat", "style": "Right-arm offbreak", "role": "Bowling All-rounder"},
    "B Kumar": {"hand": "Right-hand bat", "style": "Right-arm medium-fast", "role": "Bowler"},
    "Mohammed Shami": {"hand": "Right-hand bat", "style": "Right-arm fast", "role": "Bowler"},
    "Mohammed Siraj": {"hand": "Right-hand bat", "style": "Right-arm fast", "role": "Bowler"},
    "K Rabada": {"hand": "Left-hand bat", "style": "Right-arm fast", "role": "Bowler"},
    "TA Boult": {"hand": "Right-hand bat", "style": "Left-arm fast-medium", "role": "Bowler"},
    "MA Starc": {"hand": "Left-hand bat", "style": "Left-arm fast", "role": "Bowler"},
    "Kuldeep Yadav": {"hand": "Left-hand bat", "style": "Left-arm wrist spin", "role": "Bowler"},
    "A Mishra": {"hand": "Right-hand bat", "style": "Right-arm legbreak", "role": "Bowler"},
    "PP Chawla": {"hand": "Left-hand bat", "style": "Right-arm legbreak", "role": "Bowler"},
    "Harbhajan Singh": {"hand": "Right-hand bat", "style": "Right-arm offbreak", "role": "Bowler"},
    "SL Malinga": {"hand": "Right-hand bat", "style": "Right-arm fast", "role": "Bowler"},
    "UT Yadav": {"hand": "Right-hand bat", "style": "Right-arm fast", "role": "Bowler"},
    "HV Patel": {"hand": "Right-hand bat", "style": "Right-arm medium-fast", "role": "Bowler"},
    "Arshdeep Singh": {"hand": "Left-hand bat", "style": "Left-arm medium-fast", "role": "Bowler"},
    "Avesh Khan": {"hand": "Right-hand bat", "style": "Right-arm fast-medium", "role": "Bowler"},
    "CV Varun": {"hand": "Right-hand bat", "style": "Right-arm legbreak", "role": "Bowler"},
    "T Natarajan": {"hand": "Left-hand bat", "style": "Left-arm medium-fast", "role": "Bowler"},
    "JD Unadkat": {"hand": "Right-hand bat", "style": "Left-arm medium-fast", "role": "Bowler"},
    "MM Sharma": {"hand": "Right-hand bat", "style": "Right-arm medium", "role": "Bowler"},
    "Sandeep Sharma": {"hand": "Right-hand bat", "style": "Right-arm medium", "role": "Bowler"}
}

class DataTransformer:
    """Transforms raw deliveries into enriched analytical features."""

    @staticmethod
    def get_player_metadata(player_name: str) -> Dict[str, str]:
        """Returns metadata for player, defaulting sensibly if not explicitly registered."""
        if player_name in PLAYER_PROFILES:
            return PLAYER_PROFILES[player_name]
        return {
            "hand": "Right-hand bat",
            "style": "Right-arm medium",
            "role": "Player"
        }

    @staticmethod
    def enrich_deliveries(df: pd.DataFrame, matches_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Derives phases, match situation, cumulative runs/wickets, and pressure index.
        """
        df = df.copy()

        # Phase derivation
        df["phase"] = df["over"].apply(get_match_phase)

        # Merge match-level target_runs if matches_df provided and not already present
        if matches_df is not None and not matches_df.empty:
            if "target_runs" in matches_df.columns and "target_runs" not in df.columns:
                target_map = matches_df.set_index("match_id")["target_runs"].to_dict()
                df["target_runs"] = df["match_id"].map(target_map)

        if "target_runs" not in df.columns:
            df["target_runs"] = np.nan

        # Sort chronologically by match, innings, over, ball
        sort_cols = [c for c in ["match_id", "innings", "over", "ball_number"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # Calculate cumulative score and wickets per innings
        group_keys = ["match_id", "innings"]
        df["cum_runs"] = df.groupby(group_keys)["total_runs"].cumsum()
        df["cum_wickets"] = df.groupby(group_keys)["is_wicket"].cumsum()

        # Delivery sequence within innings
        df["delivery_number"] = df.groupby(group_keys).cumcount() + 1

        # Match Pressure Index calculation
        # Vectorized / batch calculation of statistical match situation pressure
        pressures = []
        for row in df[["innings", "over", "cum_runs", "cum_wickets", "target_runs"]].itertuples(index=False):
            p = calculate_pressure_index(
                innings=int(row.innings),
                over=float(row.over),
                current_runs=int(row.cum_runs),
                wickets_lost=int(row.cum_wickets),
                target_runs=int(row.target_runs) if pd.notna(row.target_runs) else None
            )
            pressures.append(p)
        df["pressure_index"] = pressures

        # Pressure category
        df["pressure_category"] = pd.cut(
            df["pressure_index"],
            bins=[0, 45, 75, 100],
            labels=["Low", "Medium", "High"]
        ).astype(str)

        # Add bowler and batter styles
        df["batter_hand"] = df["batter"].apply(lambda p: DataTransformer.get_player_metadata(p)["hand"])
        df["bowler_style"] = df["bowler"].apply(lambda p: DataTransformer.get_player_metadata(p)["style"])
        
        # Bowling category: Pace vs Spin
        from src.utils.constants import PACE_STYLES, SPIN_STYLES
        df["bowler_type"] = df["bowler_style"].apply(
            lambda s: "Spin" if s in SPIN_STYLES else ("Pace" if s in PACE_STYLES else "Pace")
        )

        return df
