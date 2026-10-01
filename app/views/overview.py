"""CricketIQ Command Center and Platform Overview Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.database.queries import CricketIQQueries
from app.components.cards import render_metric_card, render_hero_banner
from app.components.tables import render_styled_dataframe

def render():
    render_hero_banner(
        title="Command Center",
        subtitle="Real-time sports intelligence, all-time cricket delivery corpus (2008–2024), and explainable ML telemetry.",
        badge="1,243 Matches Indexed"
    )

    # Fetch global KPIs
    kpis = CricketIQQueries.get_overview_kpis()

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("Total Encounters", f"{kpis['total_matches']:,}", "All Historical Matches")
    with c2:
        render_metric_card("Deliveries Analyzed", f"{kpis['total_deliveries']:,}", "Ball-by-ball Precision")
    with c3:
        render_metric_card("Runs Logged", f"{kpis['total_runs']:,}", "Off Bat & Extras")
    with c4:
        render_metric_card("Wickets Indexed", f"{kpis['total_wickets']:,}", "Bowler Dismissals")
    with c5:
        render_metric_card("Player Profiles", f"{kpis['total_players']:,}", "Batters & Bowlers")

    st.markdown("<hr style='margin: 1.5rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

    # Macro Analytics Charts (Phase Run Share & Season Progression)
    st.subheader("📊 Macro Tournament Telemetry")
    col_phase, col_season = st.columns([1, 1.2])

    with col_phase:
        phase_df = CricketIQQueries.get_phase_distribution()
        if not phase_df.empty:
            colors = ["#3B82F6", "#10B981", "#EF4444"]
            fig_phase = go.Figure(data=[
                go.Pie(
                    labels=phase_df["phase"],
                    values=phase_df["runs"],
                    hole=0.55,
                    marker=dict(colors=colors, line=dict(color="#0B0F19", width=2)),
                    textinfo="label+percent",
                    textfont=dict(size=12, color="#F8FAFC", family="Plus Jakarta Sans"),
                    hoverinfo="label+value+percent"
                )
            ])
            fig_phase.update_layout(
                title=dict(text="Runs Distributed by Match Phase", font=dict(size=13, color="#94A3B8")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=260,
                margin=dict(l=20, r=20, t=35, b=20)
            )
            st.plotly_chart(fig_phase, use_container_width=True)

    with col_season:
        season_df = CricketIQQueries.get_season_match_counts()
        if not season_df.empty:
            fig_season = go.Figure(data=[
                go.Bar(
                    x=season_df["season"].astype(str),
                    y=season_df["matches"],
                    marker=dict(
                        color=season_df["matches"],
                        colorscale="Blues",
                        line=dict(color="#3B82F6", width=1)
                    ),
                    text=season_df["matches"],
                    textposition="auto",
                    textfont=dict(color="#F8FAFC", size=10)
                )
            ])
            fig_season.update_layout(
                title=dict(text="Historical Matches Indexed by Season", font=dict(size=13, color="#94A3B8")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#94A3B8")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10, color="#94A3B8")),
                height=260,
                margin=dict(l=20, r=20, t=35, b=20)
            )
            st.plotly_chart(fig_season, use_container_width=True)

    st.markdown("<hr style='margin: 1.5rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

    # Top Performers Row
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("🏆 All-Time Leading Run Scorers")
        top_scorers = pd.DataFrame(kpis["top_scorers"])
        if not top_scorers.empty:
            top_scorers.columns = ["Player", "Total Runs", "Balls Faced", "Strike Rate"]
            render_styled_dataframe(top_scorers)

    with col_right:
        st.subheader("🎯 All-Time Leading Wicket Takers")
        top_bowlers = pd.DataFrame(kpis["top_wicket_takers"])
        if not top_bowlers.empty:
            top_bowlers.columns = ["Player", "Wickets", "Balls Bowled", "Economy Rate"]
            render_styled_dataframe(top_bowlers)

    st.markdown("<hr style='margin: 1.5rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
    st.subheader("⚡ Analytical Architecture & Intelligence Engines")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            """
            <div class="cricketiq-card">
                <div style="font-weight: 700; color: #60A5FA; margin-bottom: 6px; font-size: 0.95rem;">🔬 Micro-Feature Engineering</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
                    • <strong>Phase Decomposition:</strong> Powerplay (1-6), Middle (7-15), Death (16-20)<br>
                    • <strong>Pace vs Spin:</strong> Split performance by bowler classification<br>
                    • <strong>Pressure Proxy:</strong> RRR leverage and required scoring rate stress<br>
                    • <strong>Consistency:</strong> Normalized CV & IQR index
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with f2:
        st.markdown(
            """
            <div class="cricketiq-card">
                <div style="font-weight: 700; color: #34D399; margin-bottom: 6px; font-size: 0.95rem;">🤖 Supervised & Unsupervised ML</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
                    • <strong>Time-Aware Splits:</strong> Chronological train/test (zero future leakage)<br>
                    • <strong>Intervals:</strong> Residual quantile 80% prediction intervals<br>
                    • <strong>Explainable AI:</strong> Exact game-theoretic TreeSHAP feature attribution<br>
                    • <strong>Clustering:</strong> K-Means role archetype manifolds
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with f3:
        st.markdown(
            """
            <div class="cricketiq-card">
                <div style="font-weight: 700; color: #FBBF24; margin-bottom: 6px; font-size: 0.95rem;">🏟️ Stadium & Counterfactual Simulation</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
                    • <strong>Venue Difficulty Index:</strong> Quantitative stadium scoring drag<br>
                    • <strong>CPI Index:</strong> Contextual Performance Index vs match par<br>
                    • <strong>What-If Simulator:</strong> Counterfactual conditions comparison<br>
                    • <strong>NetworkX:</strong> Batter-Bowler bipartite graph embeddings
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
