"""Dynamic Form Modeling and Trajectory Analysis Page."""

import streamlit as st
import pandas as pd
from src.features.form_features import FormFeatureExtractor
from app.components.cards import render_metric_card, render_trend_badge, render_hero_banner
from app.components.charts import plot_form_trajectory
from app.components.tables import render_styled_dataframe

def render(player_name: str):
    render_hero_banner(
        title=f"Form Evolution: {player_name}",
        subtitle="Time-decayed exponential form modeling, moving averages, and trajectory classification.",
        badge="Form Modeling"
    )

    # Form parameters in styled card
    st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Model Tuning Parameters</div>", unsafe_allow_html=True)
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        window = st.slider("Moving Window (Matches)", min_value=3, max_value=20, value=10)
    with col_c2:
        decay = st.slider("Exponential Decay Factor (λ)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    with col_c3:
        role = st.selectbox("Player Discipline", ["batting", "bowling"])

    # Compute dynamic form
    form_res = FormFeatureExtractor.calculate_player_form(player_name, role=role, window=window, decay=decay)

    # Metric Cards
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        render_metric_card("Form Rating", f"{form_res['recent_form_score']:.0f} / 100", f"Status: {form_res.get('trend', 'Stable')}")
    with f2:
        render_metric_card("Career Avg", f"{form_res['career_avg']:.1f}", "All historical encounters")
    with f3:
        render_metric_card(f"Recent {window} Avg", f"{form_res.get('recent_n_avg', 0.0):.1f}", f"Trailing {window} matches")
    with f4:
        render_metric_card("Trajectory Gradient", f"{form_res.get('slope', 0.0):+.2f}", "Runs/match slope")
    with f5:
        render_metric_card("Volatility (CV)", f"{form_res.get('volatility', 0.0):.2f}", "Stability indicator")

    st.markdown("<hr style='margin: 1.5rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

    # Chart
    if form_res.get("recent_scores"):
        st.plotly_chart(
            plot_form_trajectory(
                form_res["recent_scores"],
                form_res.get("recent_dates", []),
                player_name,
                form_res["career_avg"]
            ),
            use_container_width=True
        )
    else:
        st.info("No match history found for this player and role.")
