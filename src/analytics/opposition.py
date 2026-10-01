"""Opposition Analysis Engine."""

import pandas as pd
from typing import Dict, Any, List
from sqlalchemy import text
from src.database.connection import engine
from src.utils.metrics import calculate_strike_rate, calculate_batting_average, calculate_bowling_economy

class OppositionAnalyticsEngine:
    """Analyzes player performance broken down by opposition franchise."""

    @staticmethod
    def get_player_opposition_batting(player_name: str) -> pd.DataFrame:
        """Returns batting stats against each opposition team."""
        query = text("""
            SELECT 
                d.bowling_team as opposition,
                count(DISTINCT d.match_id) as matches,
                count(*) as balls,
                sum(d.batter_runs) as runs,
                sum(d.is_four) as fours,
                sum(d.is_six) as sixes,
                sum(CASE WHEN d.player_dismissed = :p THEN 1 ELSE 0 END) as dismissals
            FROM deliveries d
            WHERE d.batter = :p AND d.bowling_team IS NOT NULL
            GROUP BY d.bowling_team
            ORDER BY runs DESC
        """)
        df = pd.read_sql(query, engine, params={"p": player_name})
        if not df.empty:
            df["strike_rate"] = df.apply(lambda r: calculate_strike_rate(r["runs"], r["balls"]), axis=1)
            df["average"] = df.apply(lambda r: calculate_batting_average(r["runs"], r["dismissals"]), axis=1)
            df["boundary_pct"] = (
                (df["fours"] * 4 + df["sixes"] * 6) / df["runs"].replace(0, 1) * 100.0
            ).round(1)
        return df

    @staticmethod
    def get_player_opposition_bowling(player_name: str) -> pd.DataFrame:
        """Returns bowling stats against each opposition team."""
        query = text("""
            SELECT 
                d.batting_team as opposition,
                count(DISTINCT d.match_id) as matches,
                sum(d.is_legal) as balls,
                sum(d.total_runs) as runs_conceded,
                sum(d.is_bowler_wicket) as wickets
            FROM deliveries d
            WHERE d.bowler = :p AND d.batting_team IS NOT NULL
            GROUP BY d.batting_team
            ORDER BY wickets DESC, runs_conceded ASC
        """)
        df = pd.read_sql(query, engine, params={"p": player_name})
        if not df.empty:
            df["overs"] = (df["balls"] / 6.0).round(1)
            df["economy"] = df.apply(lambda r: calculate_bowling_economy(r["runs_conceded"], r["balls"]), axis=1)
            df["strike_rate"] = (df["balls"] / df["wickets"].replace(0, 1)).round(1)
        return df
