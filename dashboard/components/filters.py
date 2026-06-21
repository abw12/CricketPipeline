from __future__ import annotations

import streamlit as st

from dashboard.services.queries import DashboardFilters, players, seasons, teams


def sidebar_filters(
    show_teams: bool = True,
    show_players: bool = False,
    show_phases: bool = False,
    key_prefix: str = "dashboard",
) -> DashboardFilters:
    with st.sidebar:
        st.subheader("Filters")
        selected_seasons = st.multiselect("Season", seasons(), key=f"{key_prefix}_season")
        selected_teams = st.multiselect("Team", teams(), key=f"{key_prefix}_team") if show_teams else []
        selected_players = st.multiselect("Player", players(), key=f"{key_prefix}_player") if show_players else []
        selected_phases = (
            st.multiselect("Phase", ["PP", "MID", "DEATH"], key=f"{key_prefix}_phase")
            if show_phases
            else []
        )

    return DashboardFilters(
        seasons=tuple(selected_seasons),
        teams=tuple(selected_teams),
        players=tuple(selected_players),
        phases=tuple(selected_phases),
    )
