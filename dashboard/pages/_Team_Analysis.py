from __future__ import annotations

import streamlit as st

from dashboard.components.charts import bar_chart, line_chart, scatter_chart
from dashboard.components.filters import sidebar_filters
from dashboard.services.queries import filter_team_phase, team_phase_season


def render() -> None:
    st.title("Team Analysis")
    filters = sidebar_filters(show_teams=True, show_phases=True, key_prefix="team")
    phases = filter_team_phase(team_phase_season(), filters)

    tab_phase, tab_efficiency = st.tabs(["Phase Performance", "Efficiency"])
    with tab_phase:
        metric = st.selectbox(
            "Metric",
            ["runs_scored", "run_rate", "wickets_lost", "wkts_taken", "economy"],
            key="team_phase_metric",
        )
        grouped = (
            phases.groupby(["team_name", "phase"], as_index=False)[metric].mean()
            if not phases.empty and {"team_name", "phase", metric}.issubset(phases.columns)
            else phases
        )
        bar_chart(
            grouped,
            "team_name",
            metric,
            f"Team {metric.replace('_', ' ').title()} by Phase",
            "phase",
            key=f"team_phase_{metric}",
        )
        st.dataframe(_sort_phase_table(phases), use_container_width=True)

    with tab_efficiency:
        scatter_chart(
            phases,
            "run_rate",
            "economy",
            "Run Rate vs Economy",
            "phase",
            "runs_scored",
            key="team_efficiency_scatter",
        )
        trend = (
            phases.groupby(["season", "team_name"], as_index=False)
            .agg(run_rate=("run_rate", "mean"), economy=("economy", "mean"))
            .sort_values("season")
            if not phases.empty and {"season", "team_name", "run_rate", "economy"}.issubset(phases.columns)
            else phases
        )
        line_chart(trend, "season", "run_rate", "Team Run Rate Trend", "team_name", key="team_run_rate_trend")


def _sort_phase_table(phases):
    columns = [column for column in ["season", "team_name", "phase"] if column in phases.columns]
    return phases.sort_values(columns) if columns else phases
