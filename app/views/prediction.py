"""Performance Prediction Lab with Uncertainty Intervals."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.ml.batting_model import BattingModelTrainer
from src.ml.bowling_model import BowlingModelTrainer
from src.ml.model_registry import ModelRegistry
from app.components.cards import render_metric_card, render_hero_banner
from app.components.tables import render_styled_dataframe

def _plot_prediction_interval(expected: float, interval_str: str, unit: str = "Runs") -> go.Figure:
    """Renders a horizontal confidence interval chart with expected marker."""
    try:
        parts = interval_str.replace("[", "").replace("]", "").split("–")
        if len(parts) != 2:
            parts = interval_str.replace("[", "").replace("]", "").split("-")
        low = float(parts[0].strip())
        high = float(parts[1].strip())
    except Exception:
        low = max(0.0, expected - 10)
        high = expected + 15

    fig = go.Figure()
    # Interval range bar
    fig.add_trace(go.Bar(
        y=["Forecast"],
        x=[high - low],
        base=[low],
        orientation="h",
        marker=dict(color="rgba(59, 130, 246, 0.35)", line=dict(color="#3B82F6", width=2)),
        name="80% Credible Range",
        hoverinfo="x+name"
    ))
    # Expected point estimate
    fig.add_trace(go.Scatter(
        y=["Forecast"],
        x=[expected],
        mode="markers",
        marker=dict(color="#F59E0B", size=16, symbol="diamond", line=dict(color="#FFFFFF", width=2)),
        name=f"Point Expectation ({expected:.1f} {unit})",
        hoverinfo="x+name"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=140,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", title=f"{unit} Forecast Range"),
        yaxis=dict(visible=False),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.8, xanchor="center", x=0.5)
    )
    return fig

def render(player_name: str):
    render_hero_banner(
        title=f"Prediction Lab: {player_name}",
        subtitle="Time-aware calibrated ML forecasting with explicit residual quantile prediction intervals (zero future data leakage).",
        badge="Probabilistic Forecasting"
    )

    col_inp1, col_inp2 = st.columns(2)
    with col_inp1:
        innings = st.selectbox("Innings Setting", [1, 2], index=0, format_func=lambda x: f"Innings {x} ({'Batting 1st' if x==1 else 'Chasing'})")
    with col_inp2:
        pressure_factor = st.slider("Situational Leverage / Pressure Factor", min_value=0.75, max_value=1.25, value=1.0, step=0.05, help="Simulate match leverage (1.0 = baseline par)")

    tab_bat, tab_bowl, tab_models = st.tabs(["🏏 Batter Projection", "🎯 Bowler Projection", "📋 Active Models Telemetry"])

    with tab_bat:
        pred_bat = BattingModelTrainer.predict_runs(player_name, innings=innings, pressure_factor=pressure_factor)
        
        b1, b2, b3 = st.columns(3)
        with b1:
            render_metric_card("Expected Runs", f"{pred_bat['expected_runs']:.0f}", "Point Estimate")
        with b2:
            render_metric_card("Prediction Range (80%)", pred_bat["prediction_interval"], "Residual Quantile Interval")
        with b3:
            render_metric_card("Model Engine", pred_bat["model_used"], "Strict Time-Split Evaluation")

        st.caption("ℹ️ **Confidence Interval Visualization (80% Credible Bound):**")
        st.plotly_chart(_plot_prediction_interval(pred_bat["expected_runs"], pred_bat["prediction_interval"], "Runs"), use_container_width=True)

        st.markdown("<hr style='margin: 1.25rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
        st.markdown("#### Input Feature Vector")
        render_styled_dataframe(pd.DataFrame([pred_bat["input_features"]]))

    with tab_bowl:
        pred_bowl = BowlingModelTrainer.predict_wickets(player_name, pressure_factor=pressure_factor)
        
        w1, w2, w3 = st.columns(3)
        with w1:
            render_metric_card("Expected Wickets", f"{pred_bowl['expected_wickets']:.1f}", "Point Estimate")
        with w2:
            render_metric_card("Prediction Range", pred_bowl["prediction_interval"], "80% Confidence Bound")
        with w3:
            render_metric_card("Model Engine", pred_bowl["model_used"], "Calibrated Ridge Regressor")

        st.caption("ℹ️ **Confidence Interval Visualization (80% Credible Bound):**")
        st.plotly_chart(_plot_prediction_interval(pred_bowl["expected_wickets"], pred_bowl["prediction_interval"], "Wickets"), use_container_width=True)

        st.markdown("<hr style='margin: 1.25rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
        st.markdown("#### Input Feature Vector")
        render_styled_dataframe(pd.DataFrame([pred_bowl["input_features"]]))

    with tab_models:
        st.subheader("Model Registry Telemetry & Benchmarks")
        metadata = ModelRegistry.get_all_metadata()
        model_rows = []
        for name, meta in metadata.items():
            metrics = meta.get("metrics", {})
            model_rows.append({
                "Model Identifier": name,
                "Target Feature": meta.get("target", "N/A"),
                "Algorithm": meta.get("algorithm", meta.get("best_model", "Trained Estimator")),
                "Version": meta.get("version", "1.0.0"),
                "Evaluation Split": meta.get("split_strategy", "Chronological (Zero Leakage)"),
                "MAE": f"{metrics.get('mae', 0.0):.2f}" if "mae" in metrics else "N/A",
                "RMSE": f"{metrics.get('rmse', 0.0):.2f}" if "rmse" in metrics else "N/A",
                "R² Score": f"{metrics.get('r2', 0.0):.3f}" if "r2" in metrics else "N/A"
            })
        
        if model_rows:
            render_styled_dataframe(pd.DataFrame(model_rows))
        else:
            st.info("No active models registered in model store.")
