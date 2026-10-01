"""Player Similarity Search Page."""

import streamlit as st
import pandas as pd
from src.ml.similarity import PlayerSimilarityEngine
from src.analytics.player_analysis import PlayerAnalyticsEngine
from app.components.cards import render_metric_card, render_hero_banner
from app.components.charts import plot_multi_radar
from app.components.tables import render_styled_dataframe

def render(player_name: str):
    render_hero_banner(
        title=f"Player Similarity: {player_name}",
        subtitle="Standardized cosine metric matching across high-dimensional feature vectors to identify statistical archetypal twins.",
        badge="Cosine Vector Matching"
    )

    col_s, _ = st.columns([1, 2])
    with col_s:
        top_n = st.slider("Number of Similar Peers to Query", min_value=2, max_value=6, value=4)
        
    similar_peers = PlayerSimilarityEngine.find_similar_players(player_name, top_n=top_n)

    if not similar_peers:
        st.warning(f"Insufficient historical data for {player_name} to compute similarity.")
        return

    st.subheader(f"Top Statistical Matches for {player_name}")

    cols = st.columns(len(similar_peers))
    for idx, peer in enumerate(similar_peers):
        with cols[idx]:
            render_metric_card(
                peer["player"],
                f"{peer['similarity_score']:.1f}%",
                f"Rank #{idx+1} Match"
            )

    st.markdown("<hr style='margin: 1.25rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

    # Detailed similarity comparison and radar overlay
    col_rad, col_tbl = st.columns([1.1, 1.2])

    with col_rad:
        top_match = similar_peers[0]["player"]
        st.subheader(f"🎯 Skill Twin Overlay: vs {top_match}")
        p1 = PlayerAnalyticsEngine.get_complete_profile(player_name)
        p2 = PlayerAnalyticsEngine.get_complete_profile(top_match)
        
        s1 = p1.get("batting_skill") if p1 and p1.get("has_batting") else (p1.get("bowling_skill") if p1 else None)
        s2 = p2.get("batting_skill") if p2 and p2.get("has_batting") else (p2.get("bowling_skill") if p2 else None)

        if s1 and s2:
            st.plotly_chart(plot_multi_radar({player_name: s1, top_match: s2}), use_container_width=True)
        else:
            st.info("Insufficient skill vectors for radar overlay.")

    with col_tbl:
        st.subheader("🔍 Comparative Trait Analysis")
        rows = []
        for peer in similar_peers:
            rows.append({
                "Player": peer["player"],
                "Similarity": f"{peer['similarity_score']:.1f}%",
                "Key Similarities": ", ".join(peer["key_similarities"]),
                "Key Differences": ", ".join(peer["key_differences"])
            })
        render_styled_dataframe(pd.DataFrame(rows))
