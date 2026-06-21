from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

from etl.common import PipelineContext, read_table, require_spark, show_preview, write_table
from etl.silver_transform.build_dim_team import normalize_team_col
from etl.silver_transform.enrich_deliveries_teams import team_keys


def enrich_matches_teams(matches: DataFrame, dim_team: DataFrame, cfg_silver: dict) -> DataFrame:
    keys = team_keys(dim_team)

    keyed = (
        matches.withColumn("team1_key", normalize_team_col(F.col("team1")))
        .withColumn("team2_key", normalize_team_col(F.col("team2")))
        .withColumn("toss_winner_key", normalize_team_col(F.col("toss_winner")))
        .withColumn("winner_key", normalize_team_col(F.col("winner")))
        .withColumn("first_key", normalize_team_col(F.col("first_batting_team")))
        .withColumn("second_key", normalize_team_col(F.col("second_batting_team")))
    )

    return (
        keyed.join(
            keys.withColumnRenamed("team_id", "team1_id").withColumnRenamed("key_norm", "team1_key"),
            "team1_key",
            "left",
        )
        .join(
            keys.withColumnRenamed("team_id", "team2_id").withColumnRenamed("key_norm", "team2_key"),
            "team2_key",
            "left",
        )
        .join(
            keys.withColumnRenamed("team_id", "toss_winner_id").withColumnRenamed(
                "key_norm", "toss_winner_key"
            ),
            "toss_winner_key",
            "left",
        )
        .join(
            keys.withColumnRenamed("team_id", "winner_id").withColumnRenamed("key_norm", "winner_key"),
            "winner_key",
            "left",
        )
        .join(
            keys.withColumnRenamed("team_id", "first_batting_team_id").withColumnRenamed(
                "key_norm", "first_key"
            ),
            "first_key",
            "left",
        )
        .join(
            keys.withColumnRenamed("team_id", "second_batting_team_id").withColumnRenamed(
                "key_norm", "second_key"
            ),
            "second_key",
            "left",
        )
        .drop("team1_key", "team2_key", "toss_winner_key", "winner_key", "first_key", "second_key")
    )


def run(
    context: Optional[PipelineContext] = None,
    spark: Optional[SparkSession] = None,
    sample_only: bool = False,
    write_out: bool = True,
    preview: bool = False,
) -> None:
    context = context or PipelineContext.load()
    spark, should_stop = require_spark("silver-enrich-matches-teams", spark)
    try:
        dim_team = read_table(spark, context.silver_table_path("dim_team"), context.silver_format)
        matches = read_table(spark, context.silver_table_path("matches"), context.silver_format)
        if sample_only:
            ids = [r["match_id"] for r in matches.select("match_id").distinct().limit(20).collect()]
            matches = matches.filter(F.col("match_id").isin(ids))

        out = enrich_matches_teams(matches, dim_team, context.silver)
        if preview:
            show_preview(
                out.select(
                    "match_id",
                    "team1",
                    "team1_id",
                    "team2",
                    "team2_id",
                    "winner",
                    "winner_id",
                ),
                "matches with team IDs",
                20,
            )
        if write_out:
            cfg = context.silver["tables"]["matches"]
            target = write_table(
                out,
                context.silver_table_path("matches"),
                context.silver_format,
                partition_columns=cfg["partition_columns"],
                repartition_columns=["season"],
                atomic=True,
            )
            print(f"Updated silver.matches with team IDs at: {target}")
    finally:
        if should_stop:
            spark.stop()


def main(write_out: bool = True, sample_only: bool = False) -> None:
    run(sample_only=sample_only, write_out=write_out and not sample_only, preview=True)


if __name__ == "__main__":
    main(write_out=True, sample_only=False)


