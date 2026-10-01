"""Multi-Player Comparison Page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.database.queries import CricketIQQueries
from src.analytics.comparison import PlayerComparisonEngine
from app.components.cards import render_hero_banner
from app.components.charts import plot_multi_radar
from app.components.tables import render_styled_dataframe

def render(default_player: str):
    render_hero_banner(
        title="Player Comparison Matrix",
        subtitle="Benchmark up to 4 cricketers across multidimensional skill vectors, phase mastery, and statistical metrics.",
        badge="Multidimensional Profiling"
    )

    all_players = CricketIQQueries.get_all_player_names()
    
    # Defaults
    defaults = [default_player]
    candidates = ["DA Warner", "KL Rahul", "RG Sharma", "AB de Villiers"]
    for c in candidates:
        if c in all_players and c not in defaults and len(defaults) < 3:
            defaults.append(c)

    selected_players = st.multiselect(
        "Select up to 4 players to compare",
        options=all_players,
        default=defaults,
        max_selections=4
    )

    if not selected_players:
        st.warning("Please select at least one player to compare.")
        return

    comp_res = PlayerComparisonEngine.compare_players(selected_players)

    # Comparison summary table
    st.subheader("📊 Comparative Statistics")
    render_styled_dataframe(comp_res["summary_df"])

    st.markdown("<hr style='margin: 1.5rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

    col_radar, col_bars = st.columns([1.1, 1.2])
    with col_radar:
        st.subheader("🎯 Skill Radar Overlay")
        if comp_res["radar_data"]:
            st.plotly_chart(plot_multi_radar(comp_res["radar_data"]), use_container_width=True)
        else:
            st.info("Insufficient data to plot skill radar.")

    with col_bars:
        st.subheader("📈 Side-by-Side Performance Metric Bars")
        sdf = comp_res["summary_df"]
        
        # Plotly grouped bar chart
        metrics_to_plot = ["Bat Avg", "Strike Rate", "Consistency"]
        colors = ["#3B82F6", "#10B981", "#F59E0B", "#EC4899"]
        
        fig = go.Figure()
        for idx, row in sdf.iterrows():
            player_name = row["Player"]
            color = colors[idx % len(colors)]
            fig.add_trace(go.Bar(
                name=player_name,
                x=metrics_to_plot,
                y=[row.get("Bat Avg", 0), row.get("Strike Rate", 0), row.get("Consistency", 0)],
                marker_color=color,
                text=[f"{row.get(m, 0):.1f}" for m in metrics_to_plot],
                textposition="auto",
                textfont=dict(size=11, color="#FFFFFF")
            ))
            
        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=390,
            margin=dict(l=20, r=20, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showgrid=True),
            xaxis=dict(tickfont=dict(size=12, color="#E2E8F0"))
        )
        st.plotly_chart(fig, use_container_width=True)
