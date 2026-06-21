from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    key: str | None = None,
) -> None:
    if _empty(df, [x, y]):
        st.info(f"No data available for {title}.")
        return
    color_arg = _optional_column(df, color)
    fig = px.bar(df, x=x, y=y, color=color_arg, title=title)
    fig.update_layout(margin=dict(l=8, r=8, t=44, b=8), xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key=key or _chart_key(title, x, y))


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    key: str | None = None,
) -> None:
    if _empty(df, [x, y]):
        st.info(f"No data available for {title}.")
        return
    color_arg = _optional_column(df, color)
    fig = px.line(df, x=x, y=y, color=color_arg, markers=True, title=title)
    fig.update_layout(margin=dict(l=8, r=8, t=44, b=8), xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key=key or _chart_key(title, x, y))


def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    size: str | None = None,
    key: str | None = None,
) -> None:
    if _empty(df, [x, y]):
        st.info(f"No data available for {title}.")
        return
    color_arg = _optional_column(df, color)
    size_arg = _optional_column(df, size)
    chart_df = _clean_scatter_data(df, x, y, size_arg)
    if chart_df.empty:
        st.info(f"No plottable data available for {title}.")
        return
    hover_data = _hover_columns(chart_df.columns, [x, y, color_arg, size_arg])
    fig = px.scatter(chart_df, x=x, y=y, color=color_arg, size=size_arg, hover_data=hover_data, title=title)
    fig.update_layout(margin=dict(l=8, r=8, t=44, b=8))
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key=key or _chart_key(title, x, y))


def _empty(df: pd.DataFrame, columns: list[str]) -> bool:
    return df.empty or any(column not in df.columns for column in columns)


def _optional_column(df: pd.DataFrame, column: str | None) -> str | None:
    return column if column and column in df.columns else None


def _clean_scatter_data(df: pd.DataFrame, x: str, y: str, size: str | None) -> pd.DataFrame:
    chart_df = df.copy()
    chart_df[x] = pd.to_numeric(chart_df[x], errors="coerce")
    chart_df[y] = pd.to_numeric(chart_df[y], errors="coerce")
    chart_df = chart_df.dropna(subset=[x, y])
    if size and size in chart_df.columns:
        chart_df[size] = pd.to_numeric(chart_df[size], errors="coerce").fillna(0).clip(lower=0)
    return chart_df


def _hover_columns(columns: Iterable[str], priority: list[str | None]) -> list[str]:
    ordered = [column for column in priority if column]
    ordered.extend(["player_name", "team_name", "season", "phase"])
    return [column for column in dict.fromkeys(ordered) if column in columns]


def _chart_key(title: str, x: str, y: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", f"{title}_{x}_{y}".lower()).strip("_")
    return f"chart_{value}"
