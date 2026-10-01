"""Multi-player comparison engine."""

import pandas as pd
from typing import List, Dict, Any
from src.analytics.player_analysis import PlayerAnalyticsEngine

class PlayerComparisonEngine:
    """Compares multiple cricket players across skill, form, phases, and consistency."""

    @staticmethod
    def compare_players(player_names: List[str]) -> Dict[str, Any]:
        """Runs side-by-side comparison for up to 4 players."""
        profiles = []
        for name in player_names[:4]:
            p = PlayerAnalyticsEngine.get_complete_profile(name)
            profiles.append(p)

        # Build comparison summary table
        rows = []
        for p in profiles:
            bs = p.get("batting_stats", {})
            bw = p.get("bowling_stats", {})
            fm = p.get("form", {})
            rows.append({
                "Player": p["name"],
                "Role": p["role"],
                "Hand": p["batting_hand"],
                "Matches": bs.get("innings", bw.get("matches", 0)),
                "Runs": bs.get("total_runs", 0),
                "Bat Avg": bs.get("batting_average", 0.0),
                "Strike Rate": bs.get("strike_rate", 0.0),
                "Wickets": bw.get("wickets", 0),
                "Economy": bw.get("economy", 0.0),
                "Recent Form": fm.get("recent_form_score", 50.0),
                "Trend": fm.get("trend", "Stable"),
                "Consistency": bs.get("consistency_score", bw.get("wkt_consistency_score", 50.0))
            })
        summary_df = pd.DataFrame(rows)

        # Skill vectors for radar visualization
        radar_data = {}
        for p in profiles:
            if p["has_batting"]:
                radar_data[p["name"]] = p["batting_skill"]
            elif p["has_bowling"]:
                radar_data[p["name"]] = p["bowling_skill"]

        return {
            "profiles": profiles,
            "summary_df": summary_df,
            "radar_data": radar_data
        }
