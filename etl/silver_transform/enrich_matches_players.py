from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

from etl.common import PipelineContext, read_table, require_spark, show_preview, write_table
from etl.silver_transform.enrich_deliveries_players import normalize_player_name, player_keys


def enrich_matches_players(matches: DataFrame, dim_player: DataFrame) -> DataFrame:
    keys = player_keys(dim_player)
    return (
        matches.withColumn("player_of_match_key", normalize_player_name(F.col("player_of_match")))
        .join(
            keys.withColumnRenamed("player_id", "player_of_match_id").withColumnRenamed(
                "player_key", "player_of_match_key"
            ),
            "player_of_match_key",
            "left",
        )
        .drop("player_of_match_key")
    )


def run(
    context: Optional[PipelineContext] = None,
    spark: Optional[SparkSession] = None,
    sample_only: bool = False,
    write_out: bool = True,
    preview: bool = False,
) -> None:
    context = context or PipelineContext.load()
    spark, should_stop = require_spark("silver-enrich-matches-players", spark)
    try:
        dim_player = read_table(spark, context.silver_table_path("dim_player"), context.silver_format)
        matches = read_table(spark, context.silver_table_path("matches"), context.silver_format)
        if sample_only:
            ids = [r["match_id"] for r in matches.select("match_id").distinct().limit(20).collect()]
            matches = matches.filter(F.col("match_id").isin(ids))

        out = enrich_matches_players(matches, dim_player)
        if preview:
            show_preview(
                out.select("match_id", "season", "player_of_match", "player_of_match_id"),
                "matches with player_of_match_id",
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
            print(f"Updated silver.matches with player_of_match_id at: {target}")
    finally:
        if should_stop:
            spark.stop()


def main(write_out: bool = True, sample_only: bool = False) -> None:
    run(sample_only=sample_only, write_out=write_out and not sample_only, preview=True)


if __name__ == "__main__":
    main(write_out=True, sample_only=False)


