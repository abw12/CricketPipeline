from __future__ import annotations

import streamlit as st


def metric_row(metrics: dict[str, str]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, value)
