"""Multidimensional Player Skill Modeling for CricketIQ."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.features.batting_features import BattingFeatureExtractor
from src.features.bowling_features import BowlingFeatureExtractor
from src.features.pressure_features import PressureFeatureExtractor

class SkillModel:
    """
    Computes normalized multidimensional skill vectors (0-100 scale) for batters and bowlers.
    All dimensions are grounded in observed cricket distributions.
    """

    @staticmethod
    def get_batting_skill_vector(player_name: str) -> Dict[str, float]:
        """Calculates 10-dimensional batting skill vector."""
        bf = BattingFeatureExtractor.extract_player_batting_features(player_name)
        if not bf or bf.get("balls_faced", 0) < 20:
            return {
                "Run Scoring": 50.0, "Strike Rotation": 50.0, "Boundary Ability": 50.0,
                "Powerplay Skill": 50.0, "Middle-Overs Skill": 50.0, "Death Skill": 50.0,
                "Pace Skill": 50.0, "Spin Skill": 50.0, "Pressure Skill": 50.0, "Consistency": 50.0
            }

        # 1. Run Scoring: Combination of Average (scaled to 45) and Total Runs
        avg_score = np.clip((bf["batting_average"] / 45.0) * 100.0, 10.0, 99.0)

        # 2. Strike Rotation: Low dot % and high scoring frequency
        # Dot % typically 25% (elite) to 50% (poor)
        rotation_score = np.clip((1.0 - (bf["dot_pct"] - 20.0) / 35.0) * 100.0, 10.0, 99.0)

        # 3. Boundary Ability: Boundary % (typically 40% to 75%) and Strike Rate
        boundary_score = np.clip(((bf["boundary_pct"] - 30.0) / 45.0) * 100.0, 10.0, 99.0)

        # 4. Phase Skills (based on Strike Rates: PP ~130, Middle ~125, Death ~170)
        pp_sr = bf.get("powerplay_strike_rate", 120.0)
        mid_sr = bf.get("middle_strike_rate", 120.0)
        death_sr = bf.get("death_strike_rate", 150.0)

        pp_score = np.clip(((pp_sr - 90.0) / 60.0) * 100.0, 10.0, 99.0)
        mid_score = np.clip(((mid_sr - 90.0) / 55.0) * 100.0, 10.0, 99.0)
        death_score = np.clip(((death_sr - 110.0) / 80.0) * 100.0, 10.0, 99.0)

        # 5. Pace & Spin Skill
        sr_pace = bf.get("sr_vs_pace", 125.0)
        sr_spin = bf.get("sr_vs_spin", 120.0)
        pace_score = np.clip(((sr_pace - 95.0) / 55.0) * 100.0, 10.0, 99.0)
        spin_score = np.clip(((sr_spin - 90.0) / 55.0) * 100.0, 10.0, 99.0)

        # 6. Pressure Skill
        pf = PressureFeatureExtractor.get_player_pressure_profile(player_name)
        hp_sr = pf.get("bat_high_sr", 125.0)
        pressure_score = np.clip(((hp_sr - 90.0) / 65.0) * 100.0, 10.0, 99.0)

        # 7. Consistency
        consistency_score = bf.get("consistency_score", 50.0)

        return {
            "Run Scoring": round(float(avg_score), 1),
            "Strike Rotation": round(float(rotation_score), 1),
            "Boundary Ability": round(float(boundary_score), 1),
            "Powerplay Skill": round(float(pp_score), 1),
            "Middle-Overs Skill": round(float(mid_score), 1),
            "Death Skill": round(float(death_score), 1),
            "Pace Skill": round(float(pace_score), 1),
            "Spin Skill": round(float(spin_score), 1),
            "Pressure Skill": round(float(pressure_score), 1),
            "Consistency": round(float(consistency_score), 1)
        }

    @staticmethod
    def get_bowling_skill_vector(player_name: str) -> Dict[str, float]:
        """Calculates 10-dimensional bowling skill vector."""
        bf = BowlingFeatureExtractor.extract_player_bowling_features(player_name)
        if not bf or bf.get("legal_balls", 0) < 30:
            return {
                "Wicket Taking": 50.0, "Economy": 50.0, "Dot Ball Skill": 50.0,
                "Powerplay Skill": 50.0, "Middle-Overs Skill": 50.0, "Death Skill": 50.0,
                "RHB Mastery": 50.0, "LHB Mastery": 50.0, "Pressure Skill": 50.0, "Consistency": 50.0
            }

        # 1. Wicket Taking: Strike Rate (lower is better, 12-25 balls/wkt)
        sr = bf["strike_rate"] if bf["strike_rate"] > 0 else 24.0
        wkt_score = np.clip(((30.0 - sr) / 16.0) * 100.0, 10.0, 99.0)

        # 2. Economy: Lower is better (6.0 is elite, 9.5 is poor)
        econ = bf["economy"] if bf["economy"] > 0 else 8.2
        econ_score = np.clip(((10.0 - econ) / 4.0) * 100.0, 10.0, 99.0)

        # 3. Dot Ball Skill: Dot % (35% to 50%)
        dot_score = np.clip(((bf["dot_pct"] - 25.0) / 25.0) * 100.0, 10.0, 99.0)

        # 4. Phase Skills (PP econ, Mid econ, Death econ)
        pp_econ = bf.get("powerplay_economy", 7.5) or 7.5
        mid_econ = bf.get("middle_economy", 7.8) or 7.8
        death_econ = bf.get("death_economy", 9.8) or 9.8

        pp_skill = np.clip(((9.5 - pp_econ) / 3.5) * 100.0, 10.0, 99.0)
        mid_skill = np.clip(((9.5 - mid_econ) / 3.5) * 100.0, 10.0, 99.0)
        death_skill = np.clip(((12.5 - death_econ) / 5.0) * 100.0, 10.0, 99.0)

        # 5. Hand Mastery
        rhb_econ = bf.get("economy_vs_rhb", 8.0) or 8.0
        lhb_econ = bf.get("economy_vs_lhb", 8.0) or 8.0
        rhb_skill = np.clip(((10.0 - rhb_econ) / 4.0) * 100.0, 10.0, 99.0)
        lhb_skill = np.clip(((10.0 - lhb_econ) / 4.0) * 100.0, 10.0, 99.0)

        # 6. Pressure Skill
        pf = PressureFeatureExtractor.get_player_pressure_profile(player_name)
        hp_econ = pf.get("bowl_high_econ", 8.5) or 8.5
        pressure_skill = np.clip(((11.0 - hp_econ) / 4.5) * 100.0, 10.0, 99.0)

        # 7. Consistency
        consistency_score = bf.get("wkt_consistency_score", 50.0)

        return {
            "Wicket Taking": round(float(wkt_score), 1),
            "Economy": round(float(econ_score), 1),
            "Dot Ball Skill": round(float(dot_score), 1),
            "Powerplay Skill": round(float(pp_skill), 1),
            "Middle-Overs Skill": round(float(mid_skill), 1),
            "Death Skill": round(float(death_skill), 1),
            "RHB Mastery": round(float(rhb_skill), 1),
            "LHB Mastery": round(float(lhb_skill), 1),
            "Pressure Skill": round(float(pressure_skill), 1),
            "Consistency": round(float(consistency_score), 1)
        }
