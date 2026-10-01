"""Venue Analytics and Intelligence Engine."""

import pandas as pd
from typing import Dict, Any, List
from src.database.connection import engine
from src.features.venue_features import VenueFeatureExtractor

class VenueAnalyticsEngine:
    """Provides high-level venue rankings and deep stadium profiling."""

    @staticmethod
    def get_all_venues_summary() -> pd.DataFrame:
        """Returns comparison table of all venues with difficulty indices."""
        query = """
            SELECT 
                venue,
                count(DISTINCT match_id) as matches,
                count(*) as balls,
                sum(total_runs) as total_runs,
                sum(is_bowler_wicket) as wickets,
                sum(is_four) as fours,
                sum(is_six) as sixes
            FROM deliveries
            WHERE venue IS NOT NULL
            GROUP BY venue
            HAVING matches >= 5
            ORDER BY matches DESC
        """
        df = pd.read_sql(query, engine)
        if df.empty:
            return pd.DataFrame()

        df["economy"] = (df["total_runs"] / df["balls"] * 6.0).round(2)
        df["avg_runs_per_match"] = (df["total_runs"] / df["matches"]).round(1)
        df["balls_per_wicket"] = (df["balls"] / df["wickets"].replace(0, 1)).round(1)
        df["boundary_pct"] = ((df["fours"] * 4 + df["sixes"] * 6) / df["total_runs"] * 100.0).round(1)

        # Difficulty Index
        df["difficulty_index"] = (50.0 + (8.2 - df["economy"]) * 12.0).clip(10, 95).round(1)
        return df

    @staticmethod
    def get_venue_detail(venue_name: str) -> Dict[str, Any]:
        """Returns deep profile for a specific venue."""
        return VenueFeatureExtractor.get_venue_profile(venue_name)
