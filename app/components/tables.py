"""Styled table rendering helpers for Streamlit."""

import streamlit as st
import pandas as pd
from typing import Optional

def render_styled_dataframe(
    df: pd.DataFrame,
    hide_index: bool = True,
    use_container_width: bool = True
) -> None:
    """Renders formatted dataframe with dark styling."""
    if df.empty:
        st.info("No records available to display.")
        return

    st.dataframe(
        df,
        hide_index=hide_index,
        use_container_width=use_container_width
    )
