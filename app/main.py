"""CricketIQ: Advanced Cricket Intelligence Platform.

Main Streamlit Application Entrypoint.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Streamlit Page Setup
st.set_page_config(
    page_title="CricketIQ | Cricket Intelligence Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure Streamlit default auto-generated multi-page sidebar navigation is hidden
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"], 
    [data-testid="stSidebarNavItems"], 
    [data-testid="stSidebarNavSeparator"],
    div[data-testid="stSidebarNav"], 
    ul[data-testid="stSidebarNavItems"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load CSS
css_file = BASE_DIR / "app" / "styles" / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Import sidebar and views
from app.components.sidebar import render_sidebar
from app.views import (
    overview,
    player_intelligence,
    player_comparison,
    matchup_analyzer,
    form_analysis,
    player_similarity,
    player_archetypes,
    venue_intelligence,
    opposition_analysis,
    prediction,
    explainability,
    what_if_simulator,
    data_explorer
)

def main():
    try:
        selected_page, selected_player = render_sidebar()

        # Page Routing
        if selected_page == "Command Center":
            overview.render()
        elif selected_page == "Player Intelligence":
            player_intelligence.render(selected_player)
        elif selected_page == "Player Comparison":
            player_comparison.render(selected_player)
        elif selected_page == "Matchup Analyzer":
            matchup_analyzer.render(selected_player)
        elif selected_page == "Form Analysis":
            form_analysis.render(selected_player)
        elif selected_page == "Player Similarity":
            player_similarity.render(selected_player)
        elif selected_page == "Player Archetypes":
            player_archetypes.render()
        elif selected_page == "Venue Intelligence":
            venue_intelligence.render()
        elif selected_page == "Opposition Analysis":
            opposition_analysis.render(selected_player)
        elif selected_page == "Prediction & SHAP":
            prediction.render(selected_player)
        elif selected_page == "Explainable AI":
            explainability.render(selected_player)
        elif selected_page == "What-If Simulator":
            what_if_simulator.render(selected_player)
        elif selected_page == "Data Explorer":
            data_explorer.render()

    except Exception as e:
        st.error(f"Application Notice: An operational error occurred while rendering the page: {e}")
        st.info("Please verify the database connection and model registry.")

if __name__ == "__main__":
    main()
