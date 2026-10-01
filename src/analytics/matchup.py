"""Batter vs Bowler Matchup Engine and Network Analytics."""

import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Optional
from sqlalchemy import text
from src.database.queries import CricketIQQueries
from src.database.connection import engine

class MatchupEngine:
    """Provides deep head-to-head analysis and NetworkX matchup interaction graphs."""

    @staticmethod
    def get_matchup_detail(batter: str, bowler: str) -> Dict[str, Any]:
        """Returns deep head-to-head statistics and sample warnings."""
        return CricketIQQueries.get_head_to_head_matchup(batter, bowler)

    @staticmethod
    def build_player_matchup_network(player_name: str, min_balls: int = 10) -> nx.DiGraph:
        """
        Builds a directed NetworkX graph of interactions involving the player.
        Nodes: Players
        Edges: Batter -> Bowler with weights (balls, runs, dismissals)
        """
        G = nx.DiGraph()

        # Check interactions where player is batter
        bat_q = text("""
            SELECT bowler, count(*) as balls, sum(batter_runs) as runs,
                   sum(is_bowler_wicket) as dismissals
            FROM deliveries
            WHERE batter = :p
            GROUP BY bowler
            HAVING balls >= :mb
        """)
        bat_df = pd.read_sql(bat_q, engine, params={"p": player_name, "mb": min_balls})
        for _, r in bat_df.iterrows():
            G.add_node(player_name, role="Batter")
            G.add_node(r["bowler"], role="Bowler")
            G.add_edge(
                player_name, r["bowler"],
                balls=int(r["balls"]), runs=int(r["runs"]), dismissals=int(r["dismissals"])
            )

        # Check interactions where player is bowler
        bowl_q = text("""
            SELECT batter, count(*) as balls, sum(batter_runs) as runs,
                   sum(is_bowler_wicket) as dismissals
            FROM deliveries
            WHERE bowler = :p
            GROUP BY batter
            HAVING balls >= :mb
        """)
        bowl_df = pd.read_sql(bowl_q, engine, params={"p": player_name, "mb": min_balls})
        for _, r in bowl_df.iterrows():
            G.add_node(r["batter"], role="Batter")
            G.add_node(player_name, role="Bowler")
            G.add_edge(
                r["batter"], player_name,
                balls=int(r["balls"]), runs=int(r["runs"]), dismissals=int(r["dismissals"])
            )

        return G
