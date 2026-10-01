"""Standardized filter widgets for analytics pages."""

import streamlit as st
from typing import List, Optional, Tuple
from src.database.queries import CricketIQQueries

def render_venue_filter(key: str = "venue_filter") -> str:
    """Renders stadium venue selection dropdown."""
    venues = CricketIQQueries.get_all_venues()
    return st.selectbox("Select Venue", venues, key=key)

def render_opposition_filter(key: str = "opp_filter") -> str:
    """Renders opposition team filter dropdown."""
    teams = CricketIQQueries.get_all_teams()
    return st.selectbox("Select Opposition", teams, key=key)
