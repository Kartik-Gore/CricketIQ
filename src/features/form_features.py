"""Dynamic Form Modeling Engine for CricketIQ."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.database.connection import engine
from sqlalchemy import text
from src.config import FORM_WINDOW, FORM_DECAY
from src.utils.metrics import calculate_exponential_weights, calculate_consistency_score
from src.utils.constants import TREND_IMPROVING, TREND_STABLE, TREND_DECLINING, TREND_VOLATILE

class FormFeatureExtractor:
    """Computes time-decayed recent form, moving averages, and statistical trend trajectories."""

    @staticmethod
    def calculate_player_form(
        player_name: str,
        role: str = "batting",
        window: int = FORM_WINDOW,
        decay: float = FORM_DECAY
    ) -> Dict[str, Any]:
        """
        Calculates dynamic recent form score and trajectory.
        
        Formula:
        weights = exp(-decay * match_age)
        form_score = sum(weights * scores) / sum(weights)
        """
        metric_col = "bat_runs" if role == "batting" else "wickets"
        filter_col = "balls_faced" if role == "batting" else "balls_bowled"

        query = text(f"""
            SELECT match_id, date, {metric_col} as score, strike_rate, economy
            FROM player_match_stats
            WHERE player = :p AND {filter_col} > 0
            ORDER BY date ASC, match_id ASC
        """)
        df = pd.read_sql(query, engine, params={"p": player_name})

        if df.empty:
            return {
                "player": player_name,
                "role": role,
                "career_avg": 0.0,
                "recent_form_score": 0.0,
                "trend": TREND_STABLE,
                "volatility": 0.0,
                "recent_scores": [],
                "recent_dates": []
            }

        scores = df["score"].values
        career_avg = round(float(np.mean(scores)), 2)

        # Recent slice
        recent_df = df.tail(window)
        recent_scores = recent_df["score"].values
        n_recent = len(recent_scores)

        if n_recent == 0:
            return {
                "player": player_name,
                "role": role,
                "career_avg": career_avg,
                "recent_form_score": career_avg,
                "trend": TREND_STABLE,
                "volatility": 0.0,
                "recent_scores": [],
                "recent_dates": []
            }

        # Exponential decay weights
        weights = calculate_exponential_weights(n_recent, decay=decay)
        weighted_form = float(np.sum(weights * recent_scores))

        # Rolling consistency / volatility
        std, cv, _ = calculate_consistency_score(recent_scores)

        # Trend trajectory calculation via linear slope
        if n_recent >= 3:
            x = np.arange(n_recent)
            slope, _ = np.polyfit(x, recent_scores, 1)
        else:
            slope = 0.0

        # Grounded trend classification
        if cv > 1.2 and n_recent >= 4:
            trend = TREND_VOLATILE
        elif slope > 1.5 or (career_avg > 0 and weighted_form > 1.20 * career_avg):
            trend = TREND_IMPROVING
        elif slope < -1.5 or (career_avg > 0 and weighted_form < 0.80 * career_avg):
            trend = TREND_DECLINING
        else:
            trend = TREND_STABLE

        # Scale form score to a 0-100 index for display
        # For batting: typical range 0-80 runs; 45 runs is elite T20 form (~90 index)
        # For bowling: typical range 0-4 wickets; 2.5 wkts is elite (~90 index)
        max_scale = 55.0 if role == "batting" else 3.0
        norm_form_index = round(float(np.clip((weighted_form / max_scale) * 100.0, 5.0, 99.0)), 1)

        return {
            "player": player_name,
            "role": role,
            "career_avg": career_avg,
            "recent_n_avg": round(float(np.mean(recent_scores)), 2),
            "exponential_form": round(weighted_form, 2),
            "recent_form_score": norm_form_index,
            "trend": trend,
            "slope": round(float(slope), 2),
            "volatility": round(float(cv), 2),
            "recent_scores": [int(s) for s in recent_scores],
            "recent_dates": recent_df["date"].tolist(),
            "matches_analyzed": n_recent
        }
