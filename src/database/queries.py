"""Optimized SQL and analytical queries for CricketIQ."""

import pandas as pd
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from src.database.connection import engine
from src.utils.logger import logger
from src.utils.metrics import (
    calculate_batting_average, calculate_strike_rate,
    calculate_bowling_economy, calculate_bowling_average,
    calculate_bowling_strike_rate, calculate_dot_percentage,
    calculate_boundary_percentage
)

class CricketIQQueries:
    """Analytical query engine interfacing with the relational database."""

    @staticmethod
    def get_overview_kpis() -> Dict[str, Any]:
        """Returns overall KPI metrics for command center."""
        with engine.connect() as conn:
            matches_count = conn.execute(text("SELECT count(*) FROM matches")).scalar() or 0
            deliveries_count = conn.execute(text("SELECT count(*) FROM deliveries")).scalar() or 0
            runs_count = conn.execute(text("SELECT sum(total_runs) FROM deliveries")).scalar() or 0
            wickets_count = conn.execute(text("SELECT sum(is_bowler_wicket) FROM deliveries")).scalar() or 0
            players_count = conn.execute(text("SELECT count(*) FROM players")).scalar() or 0
            
            # Top run scorers
            top_run_scorers = pd.read_sql(
                text("""
                    SELECT batter as player, sum(batter_runs) as total_runs, 
                           count(*) as balls, round(sum(batter_runs)*100.0/count(*), 2) as strike_rate
                    FROM deliveries
                    GROUP BY batter
                    ORDER BY total_runs DESC
                    LIMIT 5
                """),
                conn
            ).to_dict(orient="records")

            # Top wicket takers
            top_wicket_takers = pd.read_sql(
                text("""
                    SELECT bowler as player, sum(is_bowler_wicket) as wickets,
                           count(*) as balls, round(sum(total_runs)*6.0/count(*), 2) as economy
                    FROM deliveries
                    GROUP BY bowler
                    ORDER BY wickets DESC
                    LIMIT 5
                """),
                conn
            ).to_dict(orient="records")

        return {
            "total_matches": matches_count,
            "total_deliveries": deliveries_count,
            "total_runs": runs_count,
            "total_wickets": wickets_count,
            "total_players": players_count,
            "top_scorers": top_run_scorers,
            "top_wicket_takers": top_wicket_takers
        }

    @staticmethod
    def get_phase_distribution() -> pd.DataFrame:
        """Returns runs and wickets aggregated across Powerplay, Middle, Death phases."""
        query = text("""
            SELECT 
                phase,
                sum(total_runs) as runs,
                sum(is_bowler_wicket) as wickets,
                count(*) as balls
            FROM deliveries
            WHERE phase IN ('Powerplay', 'Middle', 'Death')
            GROUP BY phase
            ORDER BY CASE phase WHEN 'Powerplay' THEN 1 WHEN 'Middle' THEN 2 WHEN 'Death' THEN 3 ELSE 4 END
        """)
        return pd.read_sql(query, engine)

    @staticmethod
    def get_season_match_counts() -> pd.DataFrame:
        """Returns match counts grouped by season."""
        query = text("""
            SELECT season, count(*) as matches
            FROM matches
            GROUP BY season
            ORDER BY season ASC
        """)
        return pd.read_sql(query, engine)

    @staticmethod
    def get_all_player_names() -> List[str]:
        """Returns sorted list of all active players."""
        query = "SELECT DISTINCT name FROM players ORDER BY name ASC"
        df = pd.read_sql(query, engine)
        if df.empty:
            # Fallback to deliveries table
            df = pd.read_sql("SELECT DISTINCT batter as name FROM deliveries ORDER BY name ASC", engine)
        return df["name"].tolist()

    @staticmethod
    def get_all_venues() -> List[str]:
        """Returns list of distinct venue names."""
        query = "SELECT DISTINCT venue FROM matches WHERE venue IS NOT NULL ORDER BY venue ASC"
        df = pd.read_sql(query, engine)
        if df.empty:
            df = pd.read_sql("SELECT DISTINCT venue FROM deliveries WHERE venue IS NOT NULL ORDER BY venue ASC", engine)
        return df["venue"].tolist()

    @staticmethod
    def get_all_teams() -> List[str]:
        """Returns list of all distinct team names."""
        query = """
            SELECT DISTINCT team1 as team FROM matches WHERE team1 IS NOT NULL
            UNION
            SELECT DISTINCT team2 as team FROM matches WHERE team2 IS NOT NULL
            ORDER BY team ASC
        """
        df = pd.read_sql(query, engine)
        return [t for t in df["team"].tolist() if t and t != "Unknown"]

    @staticmethod
    def get_player_match_history(player_name: str) -> pd.DataFrame:
        """Retrieves chronological match-by-match statistics for a player."""
        query = text("""
            SELECT * FROM player_match_stats
            WHERE player = :p
            ORDER BY date ASC, match_id ASC
        """)
        return pd.read_sql(query, engine, params={"p": player_name})

    @staticmethod
    def get_player_career_batting(player_name: str) -> Dict[str, Any]:
        """Computes comprehensive career batting statistics for a player."""
        with engine.connect() as conn:
            res = conn.execute(
                text("""
                    SELECT 
                        count(DISTINCT match_id) as matches,
                        sum(batter_runs) as total_runs,
                        count(*) as balls_faced,
                        sum(CASE WHEN batter_runs = 4 THEN 1 ELSE 0 END) as fours,
                        sum(CASE WHEN batter_runs = 6 THEN 1 ELSE 0 END) as sixes,
                        sum(CASE WHEN total_runs = 0 AND is_legal = 1 THEN 1 ELSE 0 END) as dots,
                        sum(CASE WHEN player_dismissed = :p THEN 1 ELSE 0 END) as dismissals
                    FROM deliveries
                    WHERE batter = :p
                """),
                {"p": player_name}
            ).fetchone()

        if not res or res[2] is None or res[2] == 0:
            return {
                "player": player_name,
                "matches": 0, "runs": 0, "balls": 0, "average": 0.0,
                "strike_rate": 0.0, "fours": 0, "sixes": 0, "dots": 0,
                "dot_pct": 0.0, "boundary_pct": 0.0, "dismissals": 0
            }

        matches = res[0] or 0
        runs = res[1] or 0
        balls = res[2] or 0
        fours = res[3] or 0
        sixes = res[4] or 0
        dots = res[5] or 0
        dismissals = res[6] or 0

        avg = calculate_batting_average(runs, dismissals)
        sr = calculate_strike_rate(runs, balls)
        dot_pct = calculate_dot_percentage(dots, balls)
        boundary_runs = (fours * 4) + (sixes * 6)
        b_pct = calculate_boundary_percentage(boundary_runs, runs)

        return {
            "player": player_name,
            "matches": matches,
            "runs": runs,
            "balls": balls,
            "average": avg,
            "strike_rate": sr,
            "fours": fours,
            "sixes": sixes,
            "dots": dots,
            "dot_pct": dot_pct,
            "boundary_pct": b_pct,
            "dismissals": dismissals
        }

    @staticmethod
    def get_player_career_bowling(player_name: str) -> Dict[str, Any]:
        """Computes comprehensive career bowling statistics for a player."""
        with engine.connect() as conn:
            res = conn.execute(
                text("""
                    SELECT 
                        count(DISTINCT match_id) as matches,
                        count(*) as total_deliveries,
                        sum(is_legal) as legal_balls,
                        sum(total_runs) as runs_conceded,
                        sum(is_bowler_wicket) as wickets,
                        sum(CASE WHEN total_runs = 0 AND is_legal = 1 THEN 1 ELSE 0 END) as dots,
                        sum(CASE WHEN batter_runs IN (4, 6) THEN 1 ELSE 0 END) as boundaries_conceded
                    FROM deliveries
                    WHERE bowler = :p
                """),
                {"p": player_name}
            ).fetchone()

        if not res or res[1] is None or res[1] == 0:
            return {
                "player": player_name,
                "matches": 0, "overs": 0.0, "wickets": 0, "runs_conceded": 0,
                "economy": 0.0, "average": 0.0, "strike_rate": 0.0,
                "dot_pct": 0.0, "boundary_pct": 0.0, "legal_balls": 0
            }

        matches = res[0] or 0
        legal_balls = res[2] or 0
        runs = res[3] or 0
        wickets = res[4] or 0
        dots = res[5] or 0
        boundaries = res[6] or 0

        overs = round(legal_balls / 6.0, 1)
        econ = calculate_bowling_economy(runs, legal_balls)
        avg = calculate_bowling_average(runs, wickets)
        sr = calculate_bowling_strike_rate(legal_balls, wickets)
        dot_pct = calculate_dot_percentage(dots, legal_balls)
        b_pct = round((boundaries / max(1, legal_balls)) * 100.0, 2)

        return {
            "player": player_name,
            "matches": matches,
            "overs": overs,
            "wickets": wickets,
            "runs_conceded": runs,
            "economy": econ,
            "average": 0.0 if pd.isna(avg) else avg,
            "strike_rate": 0.0 if pd.isna(sr) else sr,
            "dot_pct": dot_pct,
            "boundary_pct": b_pct,
            "legal_balls": legal_balls
        }

    @staticmethod
    def get_batter_phase_stats(player_name: str) -> pd.DataFrame:
        """Returns batting phase breakdown: Powerplay, Middle, Death."""
        query = text("""
            SELECT 
                phase,
                count(*) as balls,
                sum(batter_runs) as runs,
                sum(CASE WHEN batter_runs = 4 THEN 1 ELSE 0 END) as fours,
                sum(CASE WHEN batter_runs = 6 THEN 1 ELSE 0 END) as sixes,
                sum(CASE WHEN total_runs = 0 AND is_legal = 1 THEN 1 ELSE 0 END) as dots,
                sum(CASE WHEN player_dismissed = :p THEN 1 ELSE 0 END) as dismissals
            FROM deliveries
            WHERE batter = :p
            GROUP BY phase
            ORDER BY CASE phase 
                WHEN 'Powerplay' THEN 1 
                WHEN 'Middle' THEN 2 
                ELSE 3 END
        """)
        df = pd.read_sql(query, engine, params={"p": player_name})
        if not df.empty:
            df["strike_rate"] = (df["runs"] / df["balls"] * 100.0).round(2)
            df["average"] = df.apply(
                lambda r: calculate_batting_average(r["runs"], r["dismissals"]), axis=1
            )
            df["dot_pct"] = (df["dots"] / df["balls"] * 100.0).round(2)
        return df

    @staticmethod
    def get_batter_vs_bowling_type(player_name: str) -> pd.DataFrame:
        """Analyzes batter performance against Pace vs Spin."""
        query = text("""
            SELECT 
                bowler_type,
                count(*) as balls,
                sum(batter_runs) as runs,
                sum(CASE WHEN batter_runs = 4 THEN 1 ELSE 0 END) as fours,
                sum(CASE WHEN batter_runs = 6 THEN 1 ELSE 0 END) as sixes,
                sum(CASE WHEN total_runs = 0 AND is_legal = 1 THEN 1 ELSE 0 END) as dots,
                sum(CASE WHEN player_dismissed = :p THEN 1 ELSE 0 END) as dismissals
            FROM deliveries
            WHERE batter = :p
            GROUP BY bowler_type
        """)
        df = pd.read_sql(query, engine, params={"p": player_name})
        if not df.empty:
            df["strike_rate"] = (df["runs"] / df["balls"] * 100.0).round(2)
            df["average"] = df.apply(
                lambda r: calculate_batting_average(r["runs"], r["dismissals"]), axis=1
            )
        return df

    @staticmethod
    def get_head_to_head_matchup(batter: str, bowler: str) -> Dict[str, Any]:
        """Calculates granular head-to-head matchup statistics between batter and bowler."""
        with engine.connect() as conn:
            res = conn.execute(
                text("""
                    SELECT 
                        count(*) as balls,
                        sum(batter_runs) as runs,
                        sum(CASE WHEN batter_runs = 4 THEN 1 ELSE 0 END) as fours,
                        sum(CASE WHEN batter_runs = 6 THEN 1 ELSE 0 END) as sixes,
                        sum(CASE WHEN total_runs = 0 AND is_legal = 1 THEN 1 ELSE 0 END) as dots,
                        sum(CASE WHEN player_dismissed = :batter AND is_bowler_wicket = 1 THEN 1 ELSE 0 END) as dismissals
                    FROM deliveries
                    WHERE batter = :batter AND bowler = :bowler
                """),
                {"batter": batter, "bowler": bowler}
            ).fetchone()

        balls = res[0] if res and res[0] else 0
        runs = res[1] if res and res[1] else 0
        fours = res[2] if res and res[2] else 0
        sixes = res[3] if res and res[3] else 0
        dots = res[4] if res and res[4] else 0
        dismissals = res[5] if res and res[5] else 0

        sr = calculate_strike_rate(runs, balls)
        avg = calculate_batting_average(runs, dismissals)
        dot_pct = calculate_dot_percentage(dots, balls)
        b_pct = calculate_boundary_percentage((fours * 4) + (sixes * 6), runs)

        is_limited_sample = balls < 12

        return {
            "batter": batter,
            "bowler": bowler,
            "balls": balls,
            "runs": runs,
            "dismissals": dismissals,
            "strike_rate": sr,
            "average": avg,
            "fours": fours,
            "sixes": sixes,
            "dots": dots,
            "dot_pct": dot_pct,
            "boundary_pct": b_pct,
            "is_limited_sample": is_limited_sample,
            "warning": "Limited historical sample — interpret cautiously." if is_limited_sample else None
        }
