"""Unified Player Intelligence Analytics Engine."""

from typing import Dict, Any, Optional
import pandas as pd
from src.database.queries import CricketIQQueries
from src.features.batting_features import BattingFeatureExtractor
from src.features.bowling_features import BowlingFeatureExtractor
from src.features.form_features import FormFeatureExtractor
from src.features.context_features import ContextualPerformanceEngine
from src.features.pressure_features import PressureFeatureExtractor
from src.features.matchup_features import MatchupFeatureExtractor
from src.analytics.skill import SkillModel
from src.data.transformation import DataTransformer

class PlayerAnalyticsEngine:
    """Consolidates complete 360-degree player intelligence profiles."""

    @staticmethod
    def get_complete_profile(player_name: str) -> Dict[str, Any]:
        """Returns unified player dossier."""
        metadata = DataTransformer.get_player_metadata(player_name)
        
        # Batting & Bowling core features
        batting_stats = BattingFeatureExtractor.extract_player_batting_features(player_name)
        bowling_stats = BowlingFeatureExtractor.extract_player_bowling_features(player_name)
        
        # Primary role determination
        has_batting = bool(batting_stats and batting_stats.get("balls_faced", 0) >= 15)
        has_bowling = bool(bowling_stats and bowling_stats.get("legal_balls", 0) >= 18)
        
        if has_batting and has_bowling:
            primary_role = "All-rounder"
        elif has_bowling:
            primary_role = "Bowler"
        else:
            primary_role = metadata.get("role", "Batter")

        # Form calculation
        form_role = "bowling" if primary_role == "Bowler" else "batting"
        form_data = FormFeatureExtractor.calculate_player_form(player_name, role=form_role)

        # Skill vectors
        batting_skill = SkillModel.get_batting_skill_vector(player_name) if has_batting else {}
        bowling_skill = SkillModel.get_bowling_skill_vector(player_name) if has_bowling else {}

        # Contextual Performance Index
        cpi_data = ContextualPerformanceEngine.calculate_player_cpi(player_name)

        # Pressure Profile
        pressure_profile = PressureFeatureExtractor.get_player_pressure_profile(player_name)

        # Phase breakdowns
        phase_breakdown = CricketIQQueries.get_batter_phase_stats(player_name)
        vs_bowling_type = CricketIQQueries.get_batter_vs_bowling_type(player_name)

        # Top nemeses and favourite bowlers
        nemeses = MatchupFeatureExtractor.get_batter_top_nemeses(player_name, limit=5)
        favourite_bowlers = MatchupFeatureExtractor.get_batter_favourite_bowlers(player_name, limit=5)

        return {
            "name": player_name,
            "role": primary_role,
            "batting_hand": metadata.get("hand", "Right-hand bat"),
            "bowling_style": metadata.get("style", "Right-arm medium"),
            "has_batting": has_batting,
            "has_bowling": has_bowling,
            "batting_stats": batting_stats,
            "bowling_stats": bowling_stats,
            "form": form_data,
            "batting_skill": batting_skill,
            "bowling_skill": bowling_skill,
            "cpi": cpi_data,
            "pressure": pressure_profile,
            "phase_stats": phase_breakdown,
            "vs_bowling_type": vs_bowling_type,
            "nemeses": nemeses,
            "favourite_bowlers": favourite_bowlers
        }
