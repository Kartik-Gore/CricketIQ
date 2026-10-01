"""Interactive What-If Scenario Simulator Page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.database.queries import CricketIQQueries
from src.simulation.what_if import ScenarioSimulator
from app.components.cards import render_metric_card, render_hero_banner
from app.components.tables import render_styled_dataframe

def render(default_player: str):
    render_hero_banner(
        title="What-If Scenario Simulator",
        subtitle="Counterfactual match simulation: project expected performance across differing venues, batting positions, and pressure levels.",
        badge="Counterfactual Engine"
    )

    venues = CricketIQQueries.get_all_venues()
    teams = CricketIQQueries.get_all_teams()
    all_players = CricketIQQueries.get_all_player_names()

    col_p, _ = st.columns([1, 1])
    with col_p:
        player = st.selectbox("Selected Batter to Simulate", all_players, index=all_players.index(default_player) if default_player in all_players else 0)

    st.markdown("<hr style='margin: 1.25rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            <div style="background: rgba(30, 58, 138, 0.25); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 0.85rem 1.15rem; margin-bottom: 0.75rem;">
                <span style="font-size: 0.95rem; font-weight: 800; color: #60A5FA;">🅰️ Scenario A Configuration</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        venue_a = st.selectbox("Venue (A)", venues, index=0, key="va")
        opp_a = st.selectbox("Opposition (A)", teams, index=0, key="oa")
        pos_a = st.slider("Batting Position (A)", 1, 7, 3, key="pa")
        press_a = st.select_slider("Pressure Level (A)", ["Low", "Medium", "High"], value="Medium", key="pra")

    with col_b:
        st.markdown(
            """
            <div style="background: rgba(180, 83, 9, 0.2); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 0.85rem 1.15rem; margin-bottom: 0.75rem;">
                <span style="font-size: 0.95rem; font-weight: 800; color: #FBBF24;">🅱️ Scenario B Configuration</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        v_idx_b = min(1, len(venues) - 1)
        venue_b = st.selectbox("Venue (B)", venues, index=v_idx_b, key="vb")
        opp_b = st.selectbox("Opposition (B)", teams, index=min(1, len(teams)-1), key="ob")
        pos_b = st.slider("Batting Position (B)", 1, 7, 3, key="pb")
        press_b = st.select_slider("Pressure Level (B)", ["Low", "Medium", "High"], value="High", key="prb")

    # Run simulations
    sim_a = ScenarioSimulator.simulate_batting_scenario(
        player_name=player, venue=venue_a, opposition=opp_a, batting_position=pos_a, pressure_level=press_a
    )
    sim_b = ScenarioSimulator.simulate_batting_scenario(
        player_name=player, venue=venue_b, opposition=opp_b, batting_position=pos_b, pressure_level=press_b
    )

    st.markdown("<hr style='margin: 1.25rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
    st.subheader("🎯 Side-by-Side Projection & Delta")

    col_res_a, col_res_b = st.columns(2)
    with col_res_a:
        ca1, ca2, ca3 = st.columns(3)
        with ca1:
            render_metric_card("Runs (A)", f"{sim_a['simulated_runs']:.0f}", f"Pos #{pos_a} | {press_a}")
        with ca2:
            render_metric_card("Strike Rate (A)", f"{sim_a['simulated_strike_rate']:.1f}", f"vs {opp_a}")
        with ca3:
            render_metric_card("Boundary % (A)", f"{sim_a['boundary_probability']:.1f}%", f"{venue_a[:16]}")

    with col_res_b:
        cb1, cb2, cb3 = st.columns(3)
        with cb1:
            render_metric_card("Runs (B)", f"{sim_b['simulated_runs']:.0f}", f"Pos #{pos_b} | {press_b}")
        with cb2:
            render_metric_card("Strike Rate (B)", f"{sim_b['simulated_strike_rate']:.1f}", f"vs {opp_b}")
        with cb3:
            render_metric_card("Boundary % (B)", f"{sim_b['boundary_probability']:.1f}%", f"{venue_b[:16]}")

    # Plot comparative chart
    col_chart, col_tbl = st.columns([1, 1.2])
    with col_chart:
        metrics = ["Projected Runs", "Strike Rate", "Boundary %"]
        vals_a = [sim_a["simulated_runs"], sim_a["simulated_strike_rate"], sim_a["boundary_probability"]]
        vals_b = [sim_b["simulated_runs"], sim_b["simulated_strike_rate"], sim_b["boundary_probability"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Scenario A",
            x=metrics,
            y=vals_a,
            marker_color="#3B82F6",
            text=[f"{v:.1f}" for v in vals_a],
            textposition="auto"
        ))
        fig.add_trace(go.Bar(
            name="Scenario B",
            x=metrics,
            y=vals_b,
            marker_color="#F59E0B",
            text=[f"{v:.1f}" for v in vals_b],
            textposition="auto"
        ))
        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=20, r=20, t=25, b=25),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_tbl:
        comp_df = ScenarioSimulator.compare_two_scenarios(sim_a, sim_b)
        render_styled_dataframe(comp_df)
        st.caption("ℹ️ **Methodology:** Projections blend trained baseline expectations with stadium friction multipliers, batting order availability, and empirical pressure leverage.")
