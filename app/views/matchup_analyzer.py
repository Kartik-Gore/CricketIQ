"""Head-to-Head Batter vs Bowler Matchup Analyzer."""

import streamlit as st
import pandas as pd
from src.database.queries import CricketIQQueries
from src.analytics.matchup import MatchupEngine
from app.components.cards import render_metric_card, render_sample_warning, render_hero_banner, render_versus_hero
from app.components.charts import plot_matchup_network
from app.components.tables import render_styled_dataframe

def render(default_player: str):
    render_hero_banner(
        title="Batter vs Bowler Matchup Analyzer",
        subtitle="Granular micro-matchup interaction dynamics, situational leverage, and NetworkX topological graph.",
        badge="Direct Encounters"
    )

    all_players = CricketIQQueries.get_all_player_names()
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        batter = st.selectbox("Select Batter", all_players, index=all_players.index(default_player) if default_player in all_players else 0)
    with col_sel2:
        # Default bowler to Jasprit Bumrah or another frontline bowler
        default_bowler = "JJ Bumrah" if "JJ Bumrah" in all_players else all_players[min(1, len(all_players)-1)]
        bowler = st.selectbox("Select Bowler", all_players, index=all_players.index(default_bowler) if default_bowler in all_players else 0)

    # Fetch matchup
    matchup = MatchupEngine.get_matchup_detail(batter, bowler)

    # Compute dominance status
    if matchup["balls"] == 0:
        dominance = "No Encounters"
    elif matchup["dismissals"] >= 2 and matchup["strike_rate"] < 120:
        dominance = f"Bowler Advantage (+{matchup['dismissals']} Wkts)"
    elif matchup["strike_rate"] > 145 and matchup["dismissals"] <= 1:
        dominance = f"Batter Advantage ({matchup['strike_rate']:.0f} SR)"
    else:
        dominance = "Contested Parity"

    # Versus Hero Display
    render_versus_hero(batter, bowler, dominance, matchup["balls"], matchup["runs"], matchup["dismissals"])

    # Show warning if limited sample size
    if matchup.get("is_limited_sample") and matchup["balls"] > 0:
        render_sample_warning(f"Only {matchup['balls']} balls recorded between {batter} and {bowler}. Historical sample is limited — interpret cautiously.")

    # Matchup KPIs
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        render_metric_card("Balls Faced", str(matchup["balls"]))
    with m2:
        render_metric_card("Runs Scored", str(matchup["runs"]))
    with m3:
        render_metric_card("Dismissals", str(matchup["dismissals"]))
    with m4:
        render_metric_card("Strike Rate", f"{matchup['strike_rate']:.1f}")
    with m5:
        render_metric_card("Average", f"{matchup['average']:.1f}")
    with m6:
        render_metric_card("Dot Ball %", f"{matchup['dot_pct']:.1f}%")

    st.markdown("<hr style='margin: 1.5rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

    # Matchup details table
    col_t1, col_net = st.columns([1, 1.2])
    with col_t1:
        st.subheader("📋 Granular Encounter Breakdown")
        df_match = pd.DataFrame([{
            "Fours": matchup["fours"],
            "Sixes": matchup["sixes"],
            "Dot Balls": matchup["dots"],
            "Boundary %": f"{matchup['boundary_pct']:.1f}%",
            "Scoring Frequency": f"{100.0 - matchup['dot_pct']:.1f}%"
        }])
        render_styled_dataframe(df_match)

    with col_net:
        st.subheader(f"🕸️ {batter} Interaction Network")
        G = MatchupEngine.build_player_matchup_network(batter, min_balls=12)
        if len(G.nodes) > 1:
            st.plotly_chart(plot_matchup_network(G, batter), use_container_width=True)
        else:
            st.info("Insufficient multi-bowler interaction data for network graph.")
