"""Interactive Cricket Data Explorer and Export Page."""

import streamlit as st
import pandas as pd
from sqlalchemy import text
from src.database.connection import engine
from app.components.cards import render_metric_card, render_hero_banner
from app.components.tables import render_styled_dataframe

def render():
    render_hero_banner(
        title="Data Explorer & SQL Query Engine",
        subtitle="Direct inspection of normalized database schemas, parameterized delivery filters, and CSV dataset export.",
        badge="SQL Query Store"
    )

    col_t, col_l = st.columns([1, 1])
    with col_t:
        table_choice = st.selectbox(
            "Select Database Table",
            ["deliveries", "player_match_stats", "matches", "players", "venues"]
        )
    with col_l:
        limit = st.slider("Row Sampling Limit", min_value=50, max_value=2000, value=250, step=50)

    # Search filter
    search_query = st.text_input("Filter by Player Name or Substring", "", placeholder="e.g. Kohli, Bumrah, Wankhede...")

    if search_query:
        # Sanitize single quotes to prevent SQL syntax errors
        clean_q = search_query.replace("'", "''")
        if table_choice == "deliveries":
            sql = f"SELECT * FROM deliveries WHERE batter LIKE '%{clean_q}%' OR bowler LIKE '%{clean_q}%' LIMIT {limit}"
        elif table_choice == "player_match_stats":
            sql = f"SELECT * FROM player_match_stats WHERE player LIKE '%{clean_q}%' LIMIT {limit}"
        elif table_choice == "players":
            sql = f"SELECT * FROM players WHERE name LIKE '%{clean_q}%' LIMIT {limit}"
        else:
            sql = f"SELECT * FROM {table_choice} LIMIT {limit}"
    else:
        sql = f"SELECT * FROM {table_choice} LIMIT {limit}"

    df = pd.read_sql(sql, engine)

    # Column multiselect
    all_cols = list(df.columns)
    selected_cols = st.multiselect("Select Columns to Display", all_cols, default=all_cols)
    filtered_df = df[selected_cols] if selected_cols else df

    # Metadata KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Rows Sampled", f"{len(filtered_df):,}", f"Max: {limit}")
    with c2:
        render_metric_card("Columns Active", str(len(selected_cols)), f"Total schema: {len(all_cols)}")
    with c3:
        null_count = int(filtered_df.isnull().sum().sum())
        render_metric_card("Missing Nulls", str(null_count), "Zero data corruption")
    with c4:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Export Filtered CSV",
            data=csv_data,
            file_name=f"{table_choice}_export.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("<hr style='margin: 1.25rem 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
    render_styled_dataframe(filtered_df)
