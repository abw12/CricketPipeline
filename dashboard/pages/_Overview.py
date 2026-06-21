from __future__ import annotations

import streamlit as st

from dashboard.components.cards import metric_row
from dashboard.components.charts import bar_chart, line_chart
from dashboard.components.filters import sidebar_filters
from dashboard.services import metrics
from dashboard.services.queries import (
    batter_season,
    bowler_season,
    filter_batting,
    filter_bowling,
    filter_matches,
    match_summary,
    top_n,
)


def render() -> None:
    st.title("IPL Analytics Overview")
    filters = sidebar_filters(show_teams=True, key_prefix="overview")

    matches = filter_matches(match_summary(), filters)
    batting = filter_batting(batter_season(), filters)
    bowling = filter_bowling(bowler_season(), filters)

    metric_row(metrics.overview_metrics(matches, batting, bowling))

    left, right = st.columns(2)
    with left:
        line_chart(metrics.season_runs(batting), "season", "runs", "Runs by Season", key="overview_runs_by_season")
    with right:
        line_chart(
            metrics.season_wickets(bowling),
            "season",
            "wkts",
            "Wickets by Season",
            key="overview_wickets_by_season",
        )

    left, right = st.columns(2)
    with left:
        bar_chart(top_n(batting, "runs", 10), "player_name", "runs", "Top Run Scorers", key="overview_top_runs")
    with right:
        bar_chart(top_n(bowling, "wkts", 10), "player_name", "wkts", "Top Wicket Takers", key="overview_top_wickets")

    line_chart(
        metrics.chase_summary(matches),
        "season",
        "chase_success_rate",
        "Chase Success Rate",
        key="overview_chase_success_rate",
    )
