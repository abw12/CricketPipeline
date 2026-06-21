from __future__ import annotations

import streamlit as st

from dashboard.components.charts import bar_chart
from dashboard.components.filters import sidebar_filters
from dashboard.services.queries import filter_matches, match_summary


def render() -> None:
    st.title("Match Explorer")
    filters = sidebar_filters(show_teams=True, key_prefix="match")
    matches = filter_matches(match_summary(), filters)

    venues = sorted(matches["venue"].dropna().unique()) if "venue" in matches.columns else []
    selected_venue = st.selectbox("Venue", ["All"] + venues, key="match_venue")
    if selected_venue != "All":
        matches = matches[matches["venue"] == selected_venue]

    result_types = sorted(matches["result_std"].dropna().unique()) if "result_std" in matches.columns else []
    selected_result = st.selectbox("Result Type", ["All"] + result_types, key="match_result_type")
    if selected_result != "All":
        matches = matches[matches["result_std"] == selected_result]

    left, right = st.columns(2)
    with left:
        winners = _wins_by_team(matches)
        bar_chart(winners.head(15), "winner", "matches", "Wins by Team", key="match_wins_by_team")
    with right:
        bar_chart(_chase_outcomes(matches), "chase_success", "matches", "Chase Outcomes", key="match_chase_outcomes")

    st.dataframe(
        _sort_matches(matches),
        use_container_width=True,
        hide_index=True,
    )


def _wins_by_team(matches):
    if matches.empty or "winner" not in matches.columns:
        return matches
    winners = matches["winner"].value_counts().reset_index()
    winners.columns = ["winner", "matches"]
    return winners


def _chase_outcomes(matches):
    if matches.empty or "chase_success" not in matches.columns:
        return matches
    chase = matches["chase_success"].value_counts(dropna=False).reset_index()
    chase.columns = ["chase_success", "matches"]
    chase["chase_success"] = chase["chase_success"].map(
        {True: "Successful", False: "Not Successful"}
    ).fillna("Unknown")
    return chase


def _sort_matches(matches):
    if matches.empty or "match_start_dt" not in matches.columns:
        return matches
    return matches.sort_values("match_start_dt", ascending=False)
