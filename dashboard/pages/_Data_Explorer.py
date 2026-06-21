from __future__ import annotations

import streamlit as st

from dashboard.services.queries import named_table, table_names


def render() -> None:
    st.title("Data Explorer")
    selected = st.selectbox("Table", table_names(), key="data_table")
    df = named_table(selected)

    limit = st.slider("Rows", 50, 5000, 500, step=50, key="data_row_limit")
    st.caption(f"{len(df):,} rows, {len(df.columns):,} columns")
    st.dataframe(df.head(limit), use_container_width=True)

    csv = df.head(limit).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download visible rows",
        csv,
        file_name=f"{selected.replace('.', '_')}.csv",
        key="data_download_visible_rows",
    )
