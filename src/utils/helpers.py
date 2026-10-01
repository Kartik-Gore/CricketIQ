"""General utility functions and helper methods."""

import re
import pandas as pd
from typing import Any, Optional

def clean_player_name(name: Any) -> str:
    """Standardizes player names by stripping whitespace and punctuation."""
    if not name or str(name).lower() == "nan":
        return "Unknown"
    return str(name).strip()

def clean_team_name(name: Any) -> str:
    """Standardizes team names using known franchise rebrandings."""
    if not name or str(name).lower() == "nan":
        return "Unknown"
    cleaned = str(name).strip()
    from src.utils.constants import TEAM_NAME_MAP
    return TEAM_NAME_MAP.get(cleaned, cleaned)

def clean_venue_name(name: Any) -> str:
    """Standardizes stadium and venue names."""
    if not name or str(name).lower() == "nan":
        return "Unknown Venue"
    val = str(name).strip()
    # Normalize common duplicates
    if "Wankhede" in val:
        return "Wankhede Stadium, Mumbai"
    if "Eden Gardens" in val:
        return "Eden Gardens, Kolkata"
    if "Chinnaswamy" in val or "M. Chinnaswamy" in val:
        return "M Chinnaswamy Stadium, Bengaluru"
    if "Chepauk" in val or "MA Chidambaram" in val:
        return "MA Chidambaram Stadium, Chepauk, Chennai"
    if "Arun Jaitley" in val or "Feroz Shah Kotla" in val:
        return "Arun Jaitley Stadium, Delhi"
    if "Narendra Modi" in val or "Motera" in val or "Sardar Patel" in val:
        return "Narendra Modi Stadium, Ahmedabad"
    if "Rajiv Gandhi" in val:
        return "Rajiv Gandhi International Stadium, Uppal, Hyderabad"
    if "PCA" in val or "Mohali" in val or "IS Bindra" in val:
        return "Punjab Cricket Association IS Bindra Stadium, Mohali"
    return val

def format_percentage(val: Optional[float], decimals: int = 1) -> str:
    """Formats float as percentage string."""
    if val is None or pd.isna(val):
        return "0.0%"
    return f"{val:.{decimals}f}%"

def get_match_phase(over: float) -> str:
    """Maps delivery over number (0-indexed) to cricket phase."""
    from src.config import PHASE_POWERPLAY_MAX, PHASE_MIDDLE_MAX
    from src.utils.constants import PHASE_POWERPLAY, PHASE_MIDDLE, PHASE_DEATH
    if over <= PHASE_POWERPLAY_MAX:
        return PHASE_POWERPLAY
    elif over <= PHASE_MIDDLE_MAX:
        return PHASE_MIDDLE
    else:
        return PHASE_DEATH
