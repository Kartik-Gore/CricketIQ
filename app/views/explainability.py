"""Explainable AI and SHAP Feature Attribution Page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.ml.batting_model import BattingModelTrainer
from src.explainability.shap_analysis import SHAPExplanationEngine
from app.components.cards import render_metric_card, render_hero_banner
from app.components.charts import plot_shap_waterfall_bars
from app.components.tables import render_styled_dataframe

def render(player_name: str):
    render_hero_banner(
        title=f"Explainable AI: {player_name}",
        subtitle="Game-theoretic SHAP TreeExplainer decomposing how each statistical feature shifts predictions from league baseline par.",
        badge="TreeSHAP Explainability"
    )

    pred = BattingModelTrainer.predict_runs(player_name)
    shap_res = SHAPExplanationEngine.explain_batting_prediction(pred["input_features"])

    if "error" in shap_res:
        st.error(f"SHAP error: {shap_res['error']}")
        return

    # Attribution KPI Summary
    base_val = shap_res.get("base_value", 0.0)
    pred_val = pred.get("expected_runs", 0.0)
    net_adj = pred_val - base_val

    k1, k2, k3 = st.columns(3)
    with k1:
        render_metric_card("League Baseline Par", f"{base_val:.1f} runs", "Expected Value E[f(x)]")
    with k2:
        render_metric_card("Net Attribution Shift", f"{net_adj:+.1f} runs", f"{'Positive Premium' if net_adj>=0 else 'Negative Drag'}")
    with k3:
        render_metric_card("Final Projected Runs", f"{pred_val:.0f} runs", "Model Output f(x)")

    st.markdown("<hr style='margin: 1.25rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
    st.subheader(f"📊 Local Attribution Waterfall: {player_name}")
    st.caption("Green bars indicate features boosting expected scoring; red bars indicate drag factors lowering the expectation.")

    # Plot Waterfall Chart
    st.plotly_chart(plot_shap_waterfall_bars(shap_res), use_container_width=True)

    st.markdown("<hr style='margin: 1.25rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.subheader("🟢 Top Positive Drivers (Scoring Boosts)")
        pos_df = pd.DataFrame(shap_res.get("top_positive", []))
        if not pos_df.empty:
            render_styled_dataframe(pos_df[["feature", "impact"]])
        else:
            st.info("No positive contributing factors.")

    with col_neg:
        st.subheader("🔴 Top Negative Drivers (Risk Factors)")
        neg_df = pd.DataFrame(shap_res.get("top_negative", []))
        if not neg_df.empty:
            render_styled_dataframe(neg_df[["feature", "impact"]])
        else:
            st.info("No negative suppressing factors.")

    st.markdown("<hr style='margin: 1.25rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
    st.subheader("🌐 Global Feature Importance (Tournament Wide)")
    global_imp = SHAPExplanationEngine.get_global_batting_importance()
    if global_imp:
        gdf = pd.DataFrame(global_imp)
        col_g1, col_g2 = st.columns([1.2, 1])
        with col_g1:
            fig_g = go.Figure(go.Bar(
                x=gdf["mean_abs_shap"],
                y=gdf["feature"].apply(lambda s: s.replace("_", " ").title()),
                orientation="h",
                marker_color="#3B82F6",
                text=[f"{v:.2f}" for v in gdf["mean_abs_shap"]],
                textposition="auto",
                textfont=dict(color="#FFFFFF", size=10)
            ))
            fig_g.update_layout(
                title=dict(text="Mean Absolute SHAP Value (Impact Magnitude)", font=dict(size=13, color="#94A3B8")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=20, r=20, t=35, b=20),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Mean |SHAP|"),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_g, use_container_width=True)
        with col_g2:
            render_styled_dataframe(gdf)
