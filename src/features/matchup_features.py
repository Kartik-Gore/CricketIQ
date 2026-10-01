"""Head-to-head and style matchup feature extraction."""

import pandas as pd
from typing import Dict, Any, List
from src.database.connection import engine
from sqlalchemy import text
from src.config import MIN_MATCHUP_SAMPLE
from src.utils.metrics import (
    calculate_strike_rate, calculate_batting_average,
    calculate_dot_percentage, calculate_boundary_percentage
)

class MatchupFeatureExtractor:
    """Calculates granular matchup metrics between batters and bowlers or bowling styles."""

    @staticmethod
    def get_batter_top_nemeses(batter_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Finds bowlers who have dismissed this batter the most times."""
        query = text("""
            SELECT bowler, count(*) as balls, sum(batter_runs) as runs,
                   sum(is_bowler_wicket) as dismissals,
                   sum(CASE WHEN total_runs = 0 AND is_legal = 1 THEN 1 ELSE 0 END) as dots
            FROM deliveries
            WHERE batter = :p
            GROUP BY bowler
            HAVING dismissals > 0
            ORDER BY dismissals DESC, balls ASC
            LIMIT :limit
        """)
        df = pd.read_sql(query, engine, params={"p": batter_name, "limit": limit})
        results = []
        for row in df.itertuples(index=False):
            sr = calculate_strike_rate(row.runs, row.balls)
            results.append({
                "bowler": row.bowler,
                "dismissals": int(row.dismissals),
                "balls": int(row.balls),
                "runs": int(row.runs),
                "strike_rate": sr,
                "dot_pct": calculate_dot_percentage(row.dots, row.balls)
            })
        return results

    @staticmethod
    def get_batter_favourite_bowlers(batter_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Finds bowlers against whom this batter has the highest strike rate (min 15 balls)."""
        query = text("""
            SELECT bowler, count(*) as balls, sum(batter_runs) as runs,
                   sum(is_bowler_wicket) as dismissals
            FROM deliveries
            WHERE batter = :p
            GROUP BY bowler
            HAVING balls >= 15
            ORDER BY (sum(batter_runs) * 1.0 / count(*)) DESC
            LIMIT :limit
        """)
        df = pd.read_sql(query, engine, params={"p": batter_name, "limit": limit})
        results = []
        for row in df.itertuples(index=False):
            results.append({
                "bowler": row.bowler,
                "balls": int(row.balls),
                "runs": int(row.runs),
                "strike_rate": calculate_strike_rate(row.runs, row.balls),
                "dismissals": int(row.dismissals)
            })
        return results
