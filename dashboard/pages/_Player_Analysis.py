from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import bar_chart, line_chart, scatter_chart
from dashboard.components.filters import sidebar_filters
from dashboard.services.queries import (
    batter_season,
    bowler_season,
    filter_batting,
    filter_bowling,
    players,
    top_n,
)


def render() -> None:
    st.title("Player Analysis")
    filters = sidebar_filters(show_teams=False, show_players=True, key_prefix="player")

    batting = filter_batting(batter_season(), filters)
    bowling = filter_bowling(bowler_season(), filters)

    tab_bat, tab_bowl, tab_compare = st.tabs(["Batting", "Bowling", "Compare"])
    with tab_bat:
        min_runs = st.slider("Minimum runs", 0, 1000, 100, key="player_min_runs")
        view = batting[batting["runs"] >= min_runs] if "runs" in batting.columns else batting
        bar_chart(
            top_n(view, "runs", 20),
            "player_name",
            "runs",
            "Batter Leaderboard",
            key="player_batter_leaderboard",
        )
        scatter_chart(
            view,
            "strike_rate",
            "average",
            "Batting Average vs Strike Rate",
            "season",
            "runs",
            key="player_batting_scatter",
        )
        st.dataframe(_sort_if_present(view, "runs"), use_container_width=True)

    with tab_bowl:
        min_wickets = st.slider("Minimum wickets", 0, 50, 5, key="player_min_wickets")
        view = bowling[bowling["wkts"] >= min_wickets] if "wkts" in bowling.columns else bowling
        bar_chart(
            top_n(view, "wkts", 20),
            "player_name",
            "wkts",
            "Bowler Leaderboard",
            key="player_bowler_leaderboard",
        )
        scatter_chart(
            view,
            "economy",
            "strike_rate",
            "Economy vs Bowling Strike Rate",
            "season",
            "wkts",
            key="player_bowling_scatter",
        )
        st.dataframe(_sort_if_present(view, "wkts"), use_container_width=True)

    with tab_compare:
        selected = st.multiselect("Compare players", players(), max_selections=4, key="player_compare_players")
        if selected:
            trend = batting[batting["player_name"].isin(selected)]
            line_chart(trend, "season", "runs", "Runs Trend", "player_name", key="player_compare_runs_trend")
            st.dataframe(_comparison_table(batting, bowling, selected), use_container_width=True)
        else:
            st.info("Select up to four players to compare.")


def _comparison_table(batting: pd.DataFrame, bowling: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    if "player_name" not in batting.columns or "player_name" not in bowling.columns:
        return pd.DataFrame()
    bat = batting[batting["player_name"].isin(selected)]
    bowl = bowling[bowling["player_name"].isin(selected)]
    bat_agg = bat.groupby("player_name", as_index=False).agg(
        runs=("runs", "sum"),
        balls_faced=("balls_faced", "sum"),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
    )
    bowl_agg = bowl.groupby("player_name", as_index=False).agg(
        wickets=("wkts", "sum"),
        balls_bowled=("balls_bowled", "sum"),
        runs_conceded=("runs_conceded", "sum"),
    )
    return bat_agg.merge(bowl_agg, on="player_name", how="outer").fillna(0)


def _sort_if_present(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    return df.sort_values(column, ascending=False)
