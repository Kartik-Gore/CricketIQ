"""Data Cleaning, Normalization and Deduplication for CricketIQ."""

import pandas as pd
import numpy as np
from src.utils.logger import logger
from src.utils.helpers import clean_player_name, clean_team_name, clean_venue_name
from src.utils.constants import BOWLER_DISMISSALS

class DataCleaner:
    """Provides robust sanitization for cricket ball-by-ball and match records."""

    @staticmethod
    def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and standardizes matches metadata."""
        if df.empty:
            return df

        df = df.copy()
        if "venue" in df.columns:
            df["venue"] = df["venue"].apply(clean_venue_name)
        if "team1" in df.columns:
            df["team1"] = df["team1"].apply(clean_team_name)
        if "team2" in df.columns:
            df["team2"] = df["team2"].apply(clean_team_name)
        if "winner" in df.columns:
            df["winner"] = df["winner"].apply(clean_team_name)
        if "toss_winner" in df.columns:
            df["toss_winner"] = df["toss_winner"].apply(clean_team_name)

        df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)
        return df

    @staticmethod
    def clean_deliveries(df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and normalizes deliveries dataframe."""
        df = df.copy()

        # Rename standard columns if present
        rename_map = {
            "striker": "batter",
            "batsman": "batter",
            "bowler": "bowler",
            "non_striker": "non_striker",
            "runs_off_bat": "batter_runs",
            "batsman_runs": "batter_runs",
            "start_date": "date",
            "wicket_type": "dismissal_kind"
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Standardize strings
        for col in ["batter", "bowler", "non_striker", "player_dismissed"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_player_name)

        for col in ["batting_team", "bowling_team"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_team_name)

        if "venue" in df.columns:
            df["venue"] = df["venue"].apply(clean_venue_name)

        # Standardize numeric extras
        for extra_col in ["wides", "noballs", "byes", "legbyes", "penalty", "extras"]:
            if extra_col in df.columns:
                df[extra_col] = pd.to_numeric(df[extra_col], errors="coerce").fillna(0).astype(int)
            else:
                df[extra_col] = 0

        # Batter runs & total runs
        if "batter_runs" in df.columns:
            df["batter_runs"] = pd.to_numeric(df["batter_runs"], errors="coerce").fillna(0).astype(int)
        else:
            df["batter_runs"] = 0

        if "total_runs" in df.columns:
            df["total_runs"] = pd.to_numeric(df["total_runs"], errors="coerce").fillna(
                df["batter_runs"] + df["extras"]
            ).astype(int)
        else:
            df["total_runs"] = df["batter_runs"] + df["extras"]

        # Parse over and ball
        if "ball" in df.columns and "over" not in df.columns:
            # In Cricsheet format, ball is 0.1, 0.2 ... 19.6
            raw_ball = pd.to_numeric(df["ball"], errors="coerce").fillna(0.1)
            df["over"] = raw_ball.astype(int)
            df["ball_number"] = np.round((raw_ball - df["over"]) * 10).astype(int)
        elif "over" in df.columns:
            df["over"] = pd.to_numeric(df["over"], errors="coerce").fillna(0).astype(int)
            if "ball" in df.columns:
                df["ball_number"] = pd.to_numeric(df["ball"], errors="coerce").fillna(1).astype(int)
            else:
                df["ball_number"] = 1

        # Dismissals and wickets
        if "dismissal_kind" not in df.columns:
            df["dismissal_kind"] = None
        else:
            df["dismissal_kind"] = df["dismissal_kind"].replace({np.nan: None, "": None})

        # Calculate binary wicket flag (1 if dismissal occurred)
        df["is_wicket"] = df["dismissal_kind"].apply(lambda x: 1 if x is not None and str(x).lower() != "none" else 0)
        # Bowler wicket (excludes run out, retired hurt, etc.)
        df["is_bowler_wicket"] = df["dismissal_kind"].apply(
            lambda x: 1 if x is not None and str(x).lower() in BOWLER_DISMISSALS else 0
        )

        # Legal delivery flag: wides and noballs are NOT legal deliveries
        df["is_legal"] = ((df["wides"] == 0) & (df["noballs"] == 0)).astype(int)

        # Boundary flags
        df["is_dot"] = ((df["total_runs"] == 0) & (df["is_legal"] == 1)).astype(int)
        df["is_four"] = (df["batter_runs"] == 4).astype(int)
        df["is_six"] = (df["batter_runs"] == 6).astype(int)
        df["is_boundary"] = (df["batter_runs"].isin([4, 6])).astype(int)

        # Remove duplicate records
        subset_cols = ["match_id", "innings", "over", "ball_number"]
        if all(c in df.columns for c in subset_cols):
            before = len(df)
            df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
            dropped = before - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped} duplicate delivery rows.")

        return df
