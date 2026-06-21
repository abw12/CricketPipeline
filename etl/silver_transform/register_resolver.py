from __future__ import annotations

import re
import unicodedata
from typing import Callable, Optional

from pyspark.sql import DataFrame, functions as F, types as T


def normalize_text(
    s: Optional[str],
    lower: bool = True,
    strip: bool = True,
    collapse_spaces: bool = True,
    remove_diacritics: bool = True,
    remove_trailing_commas: bool = True,
    remove_periods: bool = False,
) -> Optional[str]:
    if s is None:
        return None

    out = s
    if remove_diacritics:
        out = unicodedata.normalize("NFKD", out)
        out = "".join(ch for ch in out if not unicodedata.combining(ch))
    if lower:
        out = out.lower()
    if strip:
        out = out.strip()
    if remove_trailing_commas:
        out = re.sub(r",+$", "", out).strip()
    if remove_periods:
        out = out.replace(".", "")
    if collapse_spaces:
        out = " ".join(out.split())
    return out


def make_normalizer(rules: dict) -> Callable[[Optional[str]], Optional[str]]:
    def _norm(s: Optional[str]) -> Optional[str]:
        return normalize_text(
            s,
            lower=rules.get("lower", True),
            strip=rules.get("strip", True),
            collapse_spaces=rules.get("collapse_spaces", True),
            remove_diacritics=rules.get("remove_diacritics", True),
            remove_trailing_commas=rules.get("remove_trailing_commas", True),
            remove_periods=rules.get("remove_periods", False),
        )

    return _norm


def build_player_maps(
    players_df: DataFrame,
    aliases_df: DataFrame,
    col_p: dict,
    col_a: dict,
    normalizer: Callable[[Optional[str]], Optional[str]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for row in players_df.select(col_p["id"], col_p["unique_name"], col_p["name"]).collect():
        player_id = row[col_p["id"]]
        for value in (row[col_p["unique_name"]], row[col_p["name"]]):
            key = normalizer(value)
            if key:
                mapping[key] = player_id

    for row in aliases_df.select(col_a["id"], col_a["alias"]).collect():
        key = normalizer(row[col_a["alias"]])
        if key:
            mapping[key] = row[col_a["id"]]

    return mapping


def make_udf_player_resolver(player_map, normalizer):
    @F.udf(returnType=T.StringType())
    def resolve(name: Optional[str]) -> Optional[str]:
        key = normalizer(name)
        return None if key is None else player_map.value.get(key)

    return resolve


