"""Opposition Franchise Analytics Page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.analytics.opposition import OppositionAnalyticsEngine
from app.components.cards import render_metric_card, render_hero_banner
from app.components.tables import render_styled_dataframe

def render(player_name: str):
    render_hero_banner(
        title=f"Opposition Matrix: {player_name}",
        subtitle="Franchise-by-franchise breakdown of scoring volume, strike rates, bowling economies, and tactical matchups.",
        badge="Franchise Profiling"
    )

    tab_bat, tab_bowl = st.tabs(["🏏 Batting vs Opposition", "🎯 Bowling vs Opposition"])

    with tab_bat:
        bat_df = OppositionAnalyticsEngine.get_player_opposition_batting(player_name)
        if not bat_df.empty:
            top_opp = bat_df.iloc[0]
            highest_sr_opp = bat_df.sort_values(by="strike_rate", ascending=False).iloc[0]

            c1, c2, c3 = st.columns(3)
            with c1:
                render_metric_card("Favorite Opponent", str(top_opp["opposition"]), f"{top_opp['runs']} runs in {top_opp['matches']} matches")
            with c2:
                render_metric_card("Highest Strike Rate", str(highest_sr_opp["opposition"]), f"{highest_sr_opp['strike_rate']:.1f} SR ({highest_sr_opp['runs']} runs)")
            with c3:
                render_metric_card("Franchises Faced", f"{len(bat_df)} Teams", "Active & Historical Clubs")

            st.markdown("<hr style='margin: 1.25rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

            col_chart, col_tbl = st.columns([1.1, 1.2])
            with col_chart:
                fig_bat = go.Figure(go.Bar(
                    x=bat_df["runs"],
                    y=bat_df["opposition"],
                    orientation="h",
                    marker=dict(
                        color=bat_df["runs"],
                        colorscale="Blues",
                        line=dict(color="#3B82F6", width=1)
                    ),
                    text=[f"{r} runs (SR {sr:.0f})" for r, sr in zip(bat_df["runs"], bat_df["strike_rate"])],
                    textposition="auto",
                    textfont=dict(color="#FFFFFF", size=10)
                ))
                fig_bat.update_layout(
                    title=dict(text="Total Runs Accumulated by Opposition", font=dict(size=13, color="#94A3B8")),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=360,
                    margin=dict(l=20, r=20, t=35, b=20),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Total Runs"),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_bat, use_container_width=True)

            with col_tbl:
                st.subheader("📋 Granular Batting Matrix")
                render_styled_dataframe(bat_df)
        else:
            st.info(f"No batting records found for {player_name} against opposition franchises.")

    with tab_bowl:
        bowl_df = OppositionAnalyticsEngine.get_player_opposition_bowling(player_name)
        if not bowl_df.empty:
            top_bowl_opp = bowl_df.iloc[0]
            best_econ_opp = bowl_df.sort_values(by="economy", ascending=True).iloc[0]

            c1, c2, c3 = st.columns(3)
            with c1:
                render_metric_card("Top Victim Team", str(top_bowl_opp["opposition"]), f"{top_bowl_opp['wickets']} wickets ({top_bowl_opp['overs']} overs)")
            with c2:
                render_metric_card("Most Restrictive", str(best_econ_opp["opposition"]), f"{best_econ_opp['economy']:.2f} Economy")
            with c3:
                render_metric_card("Franchises Bowled", f"{len(bowl_df)} Teams", "Active & Historical Clubs")

            st.markdown("<hr style='margin: 1.25rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

            col_chart, col_tbl = st.columns([1.1, 1.2])
            with col_chart:
                fig_bowl = go.Figure(go.Bar(
                    x=bowl_df["wickets"],
                    y=bowl_df["opposition"],
                    orientation="h",
                    marker=dict(
                        color=bowl_df["wickets"],
                        colorscale="Teal",
                        line=dict(color="#10B981", width=1)
                    ),
                    text=[f"{w} wkts (Econ {ec:.2f})" for w, ec in zip(bowl_df["wickets"], bowl_df["economy"])],
                    textposition="auto",
                    textfont=dict(color="#FFFFFF", size=10)
                ))
                fig_bowl.update_layout(
                    title=dict(text="Total Wickets Claimed by Opposition", font=dict(size=13, color="#94A3B8")),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=360,
                    margin=dict(l=20, r=20, t=35, b=20),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Total Wickets"),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_bowl, use_container_width=True)

            with col_tbl:
                st.subheader("📋 Granular Bowling Matrix")
                render_styled_dataframe(bowl_df)
        else:
            st.info(f"No bowling records found for {player_name} against opposition franchises.")
