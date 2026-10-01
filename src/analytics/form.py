"""Form trajectory and trend analytics."""

import pandas as pd
from typing import Dict, Any, List
from src.features.form_features import FormFeatureExtractor

class FormAnalyticsEngine:
    """Provides high-level form queries and trajectory visual preparation."""

    @staticmethod
    def get_player_form_dossier(player_name: str, role: str = "batting") -> Dict[str, Any]:
        """Returns deep form analysis including trend, moving average, and raw recent matches."""
        return FormFeatureExtractor.calculate_player_form(player_name, role=role)
