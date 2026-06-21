from __future__ import annotations

import pandas as pd


def overview_metrics(matches: pd.DataFrame, batting: pd.DataFrame, bowling: pd.DataFrame) -> dict[str, str]:
    total_runs = int(batting["runs"].sum()) if "runs" in batting.columns and not batting.empty else 0
    total_wickets = int(bowling["wkts"].sum()) if "wkts" in bowling.columns and not bowling.empty else 0
    return {
        "Matches": f"{len(matches):,}",
        "Seasons": f"{matches['season'].nunique():,}" if "season" in matches.columns else "0",
        "Teams": _team_count(matches),
        "Runs": f"{total_runs:,}",
        "Wickets": f"{total_wickets:,}",
    }


def _team_count(matches: pd.DataFrame) -> str:
    if matches.empty:
        return "0"
    values = set()
    for col in ("team1", "team2"):
        if col in matches.columns:
            values.update(matches[col].dropna().astype(str).unique())
    return f"{len(values):,}"


def season_runs(batting: pd.DataFrame) -> pd.DataFrame:
    if batting.empty:
        return pd.DataFrame(columns=["season", "runs"])
    return batting.groupby("season", as_index=False)["runs"].sum().sort_values("season")


def season_wickets(bowling: pd.DataFrame) -> pd.DataFrame:
    if bowling.empty:
        return pd.DataFrame(columns=["season", "wkts"])
    return bowling.groupby("season", as_index=False)["wkts"].sum().sort_values("season")


def chase_summary(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty or "chase_success" not in matches.columns:
        return pd.DataFrame(columns=["season", "chase_success_rate"])
    out = (
        matches.groupby("season", as_index=False)["chase_success"]
        .mean()
        .rename(columns={"chase_success": "chase_success_rate"})
    )
    out["chase_success_rate"] = out["chase_success_rate"] * 100
    return out.sort_values("season")
