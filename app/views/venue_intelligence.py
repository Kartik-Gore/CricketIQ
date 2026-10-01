"""Venue Intelligence and Stadium Index Page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.analytics.venue import VenueAnalyticsEngine
from app.components.cards import render_metric_card, render_hero_banner
from app.components.tables import render_styled_dataframe

def render():
    render_hero_banner(
        title="Venue Intelligence & Stadium Profiling",
        subtitle="Pitch friction dynamics, Venue Difficulty Indices (VDI), and Pace vs Spin effectiveness across stadiums.",
        badge="Pitch Profiling"
    )

    # Global venue overview
    venues_df = VenueAnalyticsEngine.get_all_venues_summary()
    if venues_df.empty:
        st.warning("No venue records available.")
        return

    # Visual Stadium Difficulty Chart (Top 12)
    top_v = venues_df.head(12)
    colors = [
        "#EF4444" if v > 52 else ("#10B981" if v < 48 else "#3B82F6")
        for v in top_v["difficulty_index"]
    ]
    fig_vdi = go.Figure(go.Bar(
        x=top_v["difficulty_index"],
        y=top_v["venue"].apply(lambda x: x[:28]),
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1f}" for v in top_v["difficulty_index"]],
        textposition="auto",
        textfont=dict(size=10, color="#FFFFFF")
    ))
    fig_vdi.update_layout(
        title=dict(text="Top Stadiums by Venue Difficulty Index (VDI)", font=dict(size=13, color="#94A3B8")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis=dict(range=[30, 70], gridcolor="rgba(255,255,255,0.06)", title="VDI (Higher = Bowler Friendly, Lower = High Scoring)"),
        yaxis=dict(autorange="reversed")
    )

    col_chart, col_tbl = st.columns([1.1, 1.2])
    with col_chart:
        st.plotly_chart(fig_vdi, use_container_width=True)
    with col_tbl:
        st.subheader("📊 Complete Stadium Registry")
        render_styled_dataframe(venues_df)

    st.markdown("<hr style='margin: 1.5rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

    # Granular stadium inspector
    st.subheader("🔍 Deep Stadium Inspector")
    venue_list = venues_df["venue"].tolist()
    selected_venue = st.selectbox("Select Stadium to Inspect", venue_list)

    prof = VenueAnalyticsEngine.get_venue_detail(selected_venue)

    v1, v2, v3, v4, v5 = st.columns(5)
    with v1:
        render_metric_card("Difficulty Index", f"{prof.get('difficulty_index', 50.0):.1f} / 100", "VDI Score")
    with v2:
        render_metric_card("Avg 1st Innings", f"{prof.get('avg_first_innings', 160.0):.0f} runs", "Batting 1st Par")
    with v3:
        render_metric_card("Avg 2nd Innings", f"{prof.get('avg_second_innings', 150.0):.0f} runs", "Chasing Par")
    with v4:
        render_metric_card("Pace Economy", f"{prof.get('pace_economy', 8.2):.2f}", f"{prof.get('pace_wickets', 0)} wkts logged")
    with v5:
        render_metric_card("Spin Economy", f"{prof.get('spin_economy', 7.8):.2f}", f"{prof.get('spin_wickets', 0)} wkts logged")
