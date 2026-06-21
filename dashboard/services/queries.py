from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import streamlit as st

from dashboard.services.paths import DashboardPaths


@dataclass(frozen=True)
class DashboardFilters:
    seasons: tuple[str, ...] = ()
    teams: tuple[str, ...] = ()
    players: tuple[str, ...] = ()
    phases: tuple[str, ...] = ()


def paths() -> DashboardPaths:
    return DashboardPaths.load()


@st.cache_data(show_spinner=False)
def read_parquet_table(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def gold_table(table_name: str) -> pd.DataFrame:
    return read_parquet_table(str(paths().gold_table(table_name)))


def silver_table(table_name: str) -> pd.DataFrame:
    return read_parquet_table(str(paths().silver_table(table_name)))


def batter_season() -> pd.DataFrame:
    return gold_table("batter_season")


def bowler_season() -> pd.DataFrame:
    return gold_table("bowler_season")


def team_phase_season() -> pd.DataFrame:
    return gold_table("team_phase_season")


def match_summary() -> pd.DataFrame:
    df = gold_table("match_summary")
    if "match_start_dt" in df.columns:
        df = df.copy()
        df["match_start_dt"] = pd.to_datetime(df["match_start_dt"], errors="coerce")
    return df


def deliveries_sample(max_rows: int = 5000) -> pd.DataFrame:
    df = silver_table("deliveries")
    return df.head(max_rows)


def seasons() -> list[str]:
    values = set()
    for df in (match_summary(), batter_season(), bowler_season(), team_phase_season()):
        if "season" in df.columns and not df.empty:
            values.update(df["season"].dropna().astype(str).unique())
    return sorted(values)


def teams() -> list[str]:
    values = set()
    matches = match_summary()
    phases = team_phase_season()
    for col in ("team1", "team2", "winner", "first_batting_team", "second_batting_team"):
        if col in matches.columns:
            values.update(matches[col].dropna().astype(str).unique())
    if "team_name" in phases.columns:
        values.update(phases["team_name"].dropna().astype(str).unique())
    return sorted(values)


def players() -> list[str]:
    values = set()
    for df in (batter_season(), bowler_season()):
        if "player_name" in df.columns:
            values.update(df["player_name"].dropna().astype(str).unique())
    return sorted(values)


def filter_by_values(df: pd.DataFrame, column: str, values: Iterable[str]) -> pd.DataFrame:
    selected = tuple(values)
    if df.empty or not selected or column not in df.columns:
        return df
    return df[df[column].astype(str).isin(selected)]


def filter_matches(df: pd.DataFrame, filters: DashboardFilters) -> pd.DataFrame:
    out = filter_by_values(df, "season", filters.seasons)
    if filters.teams:
        team_cols = [col for col in ("team1", "team2", "winner") if col in out.columns]
        if team_cols:
            mask = False
            for col in team_cols:
                mask = mask | out[col].astype(str).isin(filters.teams)
            out = out[mask]
    return out


def filter_batting(df: pd.DataFrame, filters: DashboardFilters) -> pd.DataFrame:
    out = filter_by_values(df, "season", filters.seasons)
    out = filter_by_values(out, "player_name", filters.players)
    return out


def filter_bowling(df: pd.DataFrame, filters: DashboardFilters) -> pd.DataFrame:
    out = filter_by_values(df, "season", filters.seasons)
    out = filter_by_values(out, "player_name", filters.players)
    return out


def filter_team_phase(df: pd.DataFrame, filters: DashboardFilters) -> pd.DataFrame:
    out = filter_by_values(df, "season", filters.seasons)
    out = filter_by_values(out, "team_name", filters.teams)
    out = filter_by_values(out, "phase", filters.phases)
    return out


def top_n(df: pd.DataFrame, metric: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    return df.sort_values(metric, ascending=ascending).head(n)


def table_names() -> list[str]:
    return [
        "gold.batter_season",
        "gold.bowler_season",
        "gold.team_phase_season",
        "gold.match_summary",
        "silver.matches",
        "silver.deliveries_sample",
        "silver.dim_player",
        "silver.dim_team",
    ]


def named_table(name: str) -> pd.DataFrame:
    if name == "gold.batter_season":
        return batter_season()
    if name == "gold.bowler_season":
        return bowler_season()
    if name == "gold.team_phase_season":
        return team_phase_season()
    if name == "gold.match_summary":
        return match_summary()
    if name == "silver.deliveries_sample":
        return deliveries_sample()
    if name.startswith("silver."):
        return silver_table(name.split(".", 1)[1])
    return pd.DataFrame()
