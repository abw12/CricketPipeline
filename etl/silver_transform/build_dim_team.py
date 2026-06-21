from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

from etl.common import PipelineContext, read_table, require_spark, show_preview, write_table
from etl.silver_transform.register_resolver import make_normalizer


def select_team_column(df: DataFrame, col_name: str) -> DataFrame:
    return df.select(F.col(col_name).alias("team")).where(F.col(col_name).isNotNull())


def normalize_team_col(col):
    normalized = F.lower(F.trim(col))
    normalized = F.regexp_replace(normalized, r",+$", "")
    normalized = F.regexp_replace(normalized, r"\s+", " ")
    return normalized


def build_dim_team(matches: DataFrame, deliveries: DataFrame, cfg_silver: dict) -> DataFrame:
    match_teams = (
        select_team_column(matches, "team1")
        .unionByName(select_team_column(matches, "team2"))
        .unionByName(select_team_column(matches, "toss_winner"))
        .unionByName(select_team_column(matches, "winner"))
        .unionByName(select_team_column(matches, "first_batting_team"))
        .unionByName(select_team_column(matches, "second_batting_team"))
    )
    delivery_teams = select_team_column(deliveries, "batting_team").unionByName(
        select_team_column(deliveries, "bowling_team")
    )
    raw_teams = match_teams.unionByName(delivery_teams).distinct()

    normalizer = make_normalizer(cfg_silver["team_normalization"])
    overrides = {normalizer(k): v for k, v in (cfg_silver.get("team_overrides", {}) or {}).items()}
    override_items = []
    for key, value in overrides.items():
        override_items.extend([F.lit(key), F.lit(value)])
    override_map = F.create_map(*override_items) if override_items else F.create_map()

    return (
        raw_teams.withColumn("team_norm", normalize_team_col(F.col("team")))
        .withColumn("name_canonical", F.coalesce(F.element_at(override_map, F.col("team_norm")), F.col("team")))
        .withColumn("name_canonical_norm", normalize_team_col(F.col("name_canonical")))
        .groupBy("name_canonical", "name_canonical_norm")
        .agg(F.collect_set("team_norm").alias("aliases_norm"))
        .withColumn("team_id", F.concat(F.lit("team_"), F.substring(F.sha1(F.col("name_canonical_norm")), 1, 12)))
        .select("team_id", "name_canonical", "name_canonical_norm", "aliases_norm")
    )


def run(
    context: Optional[PipelineContext] = None,
    spark: Optional[SparkSession] = None,
    sample_only: bool = False,
    write_out: bool = True,
    preview: bool = False,
) -> None:
    context = context or PipelineContext.load()
    spark, should_stop = require_spark("silver-build-dim-team", spark)
    try:
        matches = read_table(spark, context.silver_table_path("matches"), context.silver_format)
        deliveries = read_table(spark, context.silver_table_path("deliveries"), context.silver_format)
        dim = build_dim_team(matches, deliveries, context.silver)
        if sample_only:
            dim = dim.limit(100)
        if preview:
            show_preview(dim.orderBy("name_canonical"), "dim_team", 20)
        if write_out:
            target = write_table(dim, context.silver_table_path("dim_team"), context.silver_format)
            print(f"Wrote dim_team to: {target}")
    finally:
        if should_stop:
            spark.stop()


def main(write_out: bool = True) -> None:
    run(write_out=write_out, preview=True)


if __name__ == "__main__":
    main(write_out=True)


