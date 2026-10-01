"""Detailed 360-Degree Player Dossier Page."""

import streamlit as st
import pandas as pd
from src.analytics.player_analysis import PlayerAnalyticsEngine
from src.ml.batting_model import BattingModelTrainer
from src.ml.bowling_model import BowlingModelTrainer
from src.explainability.shap_analysis import SHAPExplanationEngine
from app.components.cards import (
    render_metric_card, render_trend_badge, render_sample_warning,
    render_player_header
)
from app.components.charts import (
    plot_skill_radar, plot_form_trajectory, plot_phase_breakdown,
    plot_pace_vs_spin, plot_shap_waterfall_bars
)
from app.components.tables import render_styled_dataframe

def render(player_name: str):
    p = PlayerAnalyticsEngine.get_complete_profile(player_name)
    if not p:
        st.error(f"No intelligence profile found for {player_name}.")
        return

    # Executive Header
    render_player_header(
        name=p["name"],
        role=p["role"],
        hand=p["batting_hand"],
        style=p["bowling_style"],
        trend=p["form"].get("trend", "Stable")
    )

    # KPI Summary Cards
    bs = p.get("batting_stats", {})
    bw = p.get("bowling_stats", {})
    fm = p.get("form", {})

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        render_metric_card("Career Runs", f"{bs.get('total_runs', 0):,}", f"{bs.get('innings', 0)} innings")
    with k2:
        render_metric_card("Batting Avg", f"{bs.get('batting_average', 0.0):.1f}", f"SR: {bs.get('strike_rate', 0.0):.1f}")
    with k3:
        render_metric_card("Wickets Taken", f"{bw.get('wickets', 0):,}", f"{bw.get('overs', 0.0):.1f} overs")
    with k4:
        render_metric_card("Economy Rate", f"{bw.get('economy', 0.0):.2f}", f"Avg: {bw.get('bowling_average', 0.0):.1f}")
    with k5:
        render_metric_card("Form Rating", f"{fm.get('recent_form_score', 50.0):.0f} / 100", f"Decayed 5-match score")
    with k6:
        render_metric_card("Consistency", f"{bs.get('consistency_score', bw.get('wkt_consistency_score', 50.0)):.1f}", "Normalized (0-100)")

    st.markdown("<hr style='margin: 1.5rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

    # Tabs for in-depth analysis
    tab_skill, tab_phase, tab_pace_spin, tab_form, tab_nemesis, tab_predict = st.tabs([
        "🎯 Skill Profile",
        "⏱️ Phase Splits",
        "⚡ Pace vs Spin",
        "📈 Form Evolution",
        "⚔️ Matchups & Nemeses",
        "🔮 ML Prediction & SHAP"
    ])

    with tab_skill:
        col_rad, col_desc = st.columns([1.2, 1])
        with col_rad:
            skill_dict = p.get("batting_skill") if p["has_batting"] else p.get("bowling_skill")
            if skill_dict:
                st.plotly_chart(plot_skill_radar(skill_dict, f"{p['name']} Normalized Skill Profile"), use_container_width=True)
            else:
                st.info("Insufficient data for skill radar.")
        with col_desc:
            st.markdown("#### Skill Dimensions Breakdown")
            st.markdown(
                """
                - **Run Scoring:** Baseline average & scoring volume.
                - **Strike Rotation:** Efficiency in rotating strike with low dot ball %.
                - **Boundary Ability:** % of total runs accumulated via 4s and 6s.
                - **Phase Mastery:** Isolated control in Powerplay, Middle, and Death.
                - **Pressure Skill:** Statistical performance in high-pressure situations.
                - **Consistency Score:** Inverse coefficient of variation (0-100 scale).
                """
            )
            cpi = p.get("cpi", {})
            diff = cpi.get("performance_diff", 0.0)
            diff_str = f"{diff:+.1f} runs vs par"
            render_metric_card("Context Performance (CPI)", f"{cpi.get('cpi', 100.0)}", diff_str)

    with tab_phase:
        phase_df = p.get("phase_stats", pd.DataFrame())
        if not phase_df.empty:
            st.plotly_chart(plot_phase_breakdown(phase_df), use_container_width=True)
            render_styled_dataframe(phase_df)
        else:
            st.info("No phase split records found.")

    with tab_pace_spin:
        vs_df = p.get("vs_bowling_type", pd.DataFrame())
        if not vs_df.empty:
            col_p1, col_p2 = st.columns([1, 1.2])
            with col_p1:
                st.plotly_chart(plot_pace_vs_spin(vs_df), use_container_width=True)
            with col_p2:
                st.markdown("#### Performance vs Pace & Spin")
                render_styled_dataframe(vs_df)
        else:
            st.info("No bowling type breakdown records available.")

    with tab_form:
        if fm.get("recent_scores"):
            st.plotly_chart(
                plot_form_trajectory(
                    fm["recent_scores"],
                    fm.get("recent_dates", []),
                    p["name"],
                    fm.get("career_avg", 0.0)
                ),
                use_container_width=True
            )
            st.markdown(f"**Trend Analysis:** `{fm.get('trend')}` (Linear trajectory slope: `{fm.get('slope')}` runs/match, Volatility: `{fm.get('volatility')}` CV)")
        else:
            st.info("No chronological form history available.")

    with tab_nemesis:
        col_nem, col_fav = st.columns(2)
        with col_nem:
            st.markdown("#### 🎯 Top Nemeses (Most Dismissals)")
            nem = pd.DataFrame(p.get("nemeses", []))
            if not nem.empty:
                nem.columns = ["Bowler", "Dismissals", "Balls Faced", "Runs", "Strike Rate", "Dot %"]
                render_styled_dataframe(nem)
            else:
                st.info("No dismissal matchups recorded.")
        with col_fav:
            st.markdown("#### 🚀 Favourite Bowlers (Highest Strike Rate)")
            fav = pd.DataFrame(p.get("favourite_bowlers", []))
            if not fav.empty:
                fav.columns = ["Bowler", "Balls Faced", "Runs", "Strike Rate", "Dismissals"]
                render_styled_dataframe(fav)
            else:
                st.info("No qualifying favourite bowler records.")

    with tab_predict:
        st.markdown("#### 🔮 Machine Learning Performance Forecast")
        pred = BattingModelTrainer.predict_runs(p["name"])
        col_pr1, col_pr2 = st.columns([1, 1.2])
        with col_pr1:
            render_metric_card("Expected Runs", f"{pred['expected_runs']:.0f}", f"80% Interval: {pred['prediction_interval']}")
            render_metric_card("Active Model", pred["model_used"], "Strict Chronological Out-of-Time Split")
            st.caption("ℹ️ Probabilistic residual quantile forecast with zero temporal data leakage.")

        with col_pr2:
            shap_res = SHAPExplanationEngine.explain_batting_prediction(pred["input_features"])
            if "factors" in shap_res and shap_res["factors"]:
                st.plotly_chart(plot_shap_waterfall_bars(shap_res), use_container_width=True)
            else:
                st.info("SHAP attribution available after training models.")
