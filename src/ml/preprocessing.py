"""Feature Matrix Preprocessing with Strict Data Leakage Prevention."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sqlalchemy import text
from src.database.connection import engine
from src.utils.logger import logger

class MLDataPreprocessor:
    """
    Builds training and inference feature matrices.
    Enforces STRICT chronological time-aware ordering and rolling historical statistics
    computed strictly using matches BEFORE the target match to guarantee zero data leakage.
    """

    @staticmethod
    def prepare_batting_dataset() -> pd.DataFrame:
        """
        Creates historical match-by-match dataset for batting prediction.
        Target: bat_runs, strike_rate
        Features: prior career avg, prior form (last 5 matches avg), venue historical economy,
                  innings, opposition strength proxy.
        """
        query = text("""
            SELECT 
                pms.match_id, pms.player, pms.date, pms.season, pms.venue,
                pms.innings, pms.bat_runs, pms.balls_faced, pms.strike_rate, pms.is_out,
                p.batting_hand, p.player_type
            FROM player_match_stats pms
            JOIN players p ON pms.player = p.name
            WHERE pms.balls_faced > 0
            ORDER BY pms.date ASC, pms.match_id ASC
        """)
        df = pd.read_sql(query, engine)
        if df.empty:
            return pd.DataFrame()

        # Compute prior historical features per player (strictly before current match)
        # Using shift(1) to avoid data leakage
        df["prior_matches"] = df.groupby("player").cumcount()
        
        # Prior career runs and dismissals
        df["prior_runs"] = df.groupby("player")["bat_runs"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
        df["prior_balls"] = df.groupby("player")["balls_faced"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
        df["prior_outs"] = df.groupby("player")["is_out"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
        
        # Prior averages
        df["prior_bat_avg"] = np.where(df["prior_outs"] > 0, df["prior_runs"] / df["prior_outs"], df["prior_runs"]).round(2)
        df["prior_strike_rate"] = np.where(df["prior_balls"] > 0, (df["prior_runs"] / df["prior_balls"]) * 100.0, 120.0).round(2)

        # Prior 5-match rolling form (strictly prior)
        df["recent_form_5"] = df.groupby("player")["bat_runs"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        ).fillna(df["prior_bat_avg"])

        df["recent_sr_5"] = df.groupby("player")["strike_rate"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        ).fillna(df["prior_strike_rate"])

        # Filter players with at least 3 historical innings for reliable ML training
        df = df[df["prior_matches"] >= 3].reset_index(drop=True)
        return df

    @staticmethod
    def prepare_bowling_dataset() -> pd.DataFrame:
        """
        Creates historical match-by-match dataset for bowling prediction.
        Target: wickets, economy
        Features: prior career wickets, prior career economy, prior 5-match form, overs bowled.
        """
        query = text("""
            SELECT 
                pms.match_id, pms.player, pms.date, pms.season, pms.venue,
                pms.balls_bowled, pms.runs_conceded, pms.wickets, pms.economy,
                p.bowling_style, p.player_type
            FROM player_match_stats pms
            JOIN players p ON pms.player = p.name
            WHERE pms.balls_bowled >= 6
            ORDER BY pms.date ASC, pms.match_id ASC
        """)
        df = pd.read_sql(query, engine)
        if df.empty:
            return pd.DataFrame()

        df["prior_matches"] = df.groupby("player").cumcount()
        df["prior_wkts"] = df.groupby("player")["wickets"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
        df["prior_runs_conceded"] = df.groupby("player")["runs_conceded"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
        df["prior_balls"] = df.groupby("player")["balls_bowled"].transform(lambda s: s.shift(1).cumsum()).fillna(0)

        df["prior_econ"] = np.where(
            df["prior_balls"] > 0, (df["prior_runs_conceded"] / (df["prior_balls"] / 6.0)), 8.2
        ).round(2)

        df["recent_wkts_5"] = df.groupby("player")["wickets"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        ).fillna(1.0)

        df["recent_econ_5"] = df.groupby("player")["economy"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        ).fillna(df["prior_econ"])

        df = df[df["prior_matches"] >= 3].reset_index(drop=True)
        return df
