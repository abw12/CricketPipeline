from __future__ import annotations

import os
import threading

import streamlit as st

from dashboard.pages import (
    _Data_Explorer,
    _Match_Explorer,
    _Overview,
    _Player_Analysis,
    _Team_Analysis,
)


PAGES = {
    "Overview": _Overview.render,
    "Player Analysis": _Player_Analysis.render,
    "Team Analysis": _Team_Analysis.render,
    "Match Explorer": _Match_Explorer.render,
    "Data Explorer": _Data_Explorer.render,
}


def is_shutdown_enabled() -> bool:
    return os.getenv("CRICKET_DASHBOARD_ALLOW_SHUTDOWN") == "1"


def request_server_shutdown() -> None:
    threading.Timer(0.5, lambda: os._exit(0)).start()


def configure_page() -> None:
    st.set_page_config(
        page_title="Cricket Pipeline Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        [data-testid="stMetricValue"] { font-size: 1.65rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_shutdown_control() -> None:
    if is_shutdown_enabled() and st.button("Stop server", type="secondary"):
        st.warning("Stopping Streamlit server...")
        request_server_shutdown()
        st.stop()


def main() -> None:
    configure_page()
    with st.sidebar:
        st.header("Cricket Pipeline")
        page_name = st.radio("View", list(PAGES), label_visibility="collapsed")
        st.divider()
        render_shutdown_control()
    PAGES[page_name]()


if __name__ == "__main__":
    main()
