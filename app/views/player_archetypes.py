"""Player Archetypes and 2D Embeddings Page."""

import streamlit as st
import pandas as pd
from src.ml.clustering import PlayerClusteringEngine
from src.ml.embeddings import PlayerEmbeddingEngine
from app.components.cards import render_metric_card, render_hero_banner
from app.components.charts import plot_player_embeddings_2d
from app.components.tables import render_styled_dataframe

def render():
    render_hero_banner(
        title="Unsupervised Archetypes & 2D Manifolds",
        subtitle="Unsupervised dimensionality reduction (PCA) and K-Means clustering identifying intrinsic tactical cricket roles.",
        badge="Unsupervised Learning"
    )

    col_s, _ = st.columns([1, 2])
    with col_s:
        min_balls = st.slider("Minimum Historical Deliveries", min_value=20, max_value=100, value=35)

    emb_df = PlayerEmbeddingEngine.get_2d_player_embeddings(min_balls=min_balls)

    if emb_df.empty:
        st.warning("Insufficient data to compute embeddings and clusters.")
        return

    # Telemetry KPI cards
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Analyzed Players", f"{len(emb_df):,}", f">={min_balls} historical balls")
    with c2:
        num_archetypes = emb_df["archetype"].nunique()
        render_metric_card("Discovered Archetypes", str(num_archetypes), "K-Means Partitions")
    with c3:
        render_metric_card("Manifold Reduction", "PCA 2D", "Multidimensional Embedding")

    st.markdown("<hr style='margin: 1.25rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

    # 2D Plotly chart
    st.plotly_chart(plot_player_embeddings_2d(emb_df), use_container_width=True)

    st.markdown("<hr style='margin: 1.25rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

    # Archetype breakdown table and player inspector
    col_t1, col_t2 = st.columns([1.2, 1])
    with col_t1:
        st.subheader("📋 Discovered Archetype Profiles")
        archetype_counts = emb_df.groupby("archetype").agg(
            Players=("player", "count"),
            Avg_Scoring=("run_scoring", "mean"),
            Avg_Boundary=("boundary_ability", "mean"),
            Avg_Rotation=("strike_rotation", "mean"),
            Avg_Death=("death_skill", "mean")
        ).reset_index().round(1)
        render_styled_dataframe(archetype_counts)

    with col_t2:
        st.subheader("🔍 Archetype Roster Inspector")
        chosen_archetype = st.selectbox("Inspect Archetype", options=sorted(emb_df["archetype"].unique()))
        roster = emb_df[emb_df["archetype"] == chosen_archetype][["player", "run_scoring", "boundary_ability", "strike_rotation"]].sort_values(by="run_scoring", ascending=False)
        st.caption(f"Showing top players classified under **{chosen_archetype}**:")
        render_styled_dataframe(roster.head(15))
