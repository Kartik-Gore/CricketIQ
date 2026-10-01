"""Global Sidebar and Navigation Component."""

import streamlit as st
from typing import List, Tuple
from src.database.queries import CricketIQQueries
from src.ml.model_registry import ModelRegistry
from src.data.transformation import DataTransformer

PAGES_MAP = {
    "🏟️ Command Center": "Command Center",
    "🎯 Player Intelligence": "Player Intelligence",
    "⚖️ Player Comparison": "Player Comparison",
    "📈 Form Evolution": "Form Analysis",
    "🧬 Player Similarity": "Player Similarity",
    "🧩 Archetypes & Manifolds": "Player Archetypes",
    "⚔️ Matchup Analyzer": "Matchup Analyzer",
    "🏟️ Venue Intelligence": "Venue Intelligence",
    "🛡️ Opposition Matrix": "Opposition Analysis",
    "🔮 Prediction Lab": "Prediction & SHAP",
    "🧠 Explainable AI": "Explainable AI",
    "🎲 What-If Simulator": "What-If Simulator",
    "📂 Data Explorer": "Data Explorer"
}

def render_sidebar() -> Tuple[str, str]:
    """Renders the executive navigation sidebar and global player selector."""
    with st.sidebar:
        # Brand Header
        st.markdown(
            """
            <div style="padding: 0.25rem 0 1.25rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.6rem;">🏏</span>
                    <div>
                        <div style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.5px; color: #F8FAFC; line-height: 1.1;">CRICKET<span style="color: #3B82F6;">IQ</span></div>
                        <div style="font-size: 0.7rem; font-weight: 600; color: #64748B; letter-spacing: 0.08em; text-transform: uppercase;">Sports Intelligence</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;'>Navigation</div>", unsafe_allow_html=True)
        
        display_pages = list(PAGES_MAP.keys())
        selected_display = st.radio(
            "Navigation",
            display_pages,
            index=0,
            label_visibility="collapsed"
        )
        selected_page = PAGES_MAP[selected_display]

        st.markdown("<hr style='margin: 1.2rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

        # Active Player Selector
        st.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;'>Active Player Dossier</div>", unsafe_allow_html=True)
        all_players = CricketIQQueries.get_all_player_names()
        default_idx = 0
        if "V Kohli" in all_players:
            default_idx = all_players.index("V Kohli")

        selected_player = st.selectbox(
            "Select Player",
            options=all_players,
            index=default_idx,
            label_visibility="collapsed"
        )

        # Mini Player Info Card in Sidebar
        meta = DataTransformer.get_player_metadata(selected_player)
        role = meta.get("role", "Player")
        hand = meta.get("hand", "Right-hand bat")
        style = meta.get("style", "Right-arm medium")

        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 10px; padding: 0.75rem 0.85rem; margin-top: 0.65rem;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #F1F5F9;">{selected_player}</div>
                <div style="display: flex; gap: 6px; margin-top: 4px; flex-wrap: wrap;">
                    <span style="background: rgba(59, 130, 246, 0.2); color: #60A5FA; font-size: 0.68rem; font-weight: 600; padding: 2px 7px; border-radius: 4px;">{role}</span>
                    <span style="background: rgba(148, 163, 184, 0.15); color: #94A3B8; font-size: 0.68rem; padding: 2px 7px; border-radius: 4px;">{hand}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<hr style='margin: 1.2rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

        # System Telemetry Footer
        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #64748B; line-height: 1.5;">
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 3px;">
                    <span style="color: #10B981;">●</span> <span><strong>Corpus:</strong> 1,243 Matches (All-Time)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 3px;">
                    <span style="color: #10B981;">●</span> <span><strong>Telemetry:</strong> 295,683 Deliveries</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="color: #3B82F6;">●</span> <span><strong>AI Engine:</strong> XGBoost & SHAP</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    return selected_page, selected_player
