from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

from etl.common import PipelineContext, read_table, require_spark, show_preview, write_table


def normalize_player_name(col):
    normalized = F.lower(F.trim(col))
    normalized = F.regexp_replace(normalized, r",+$", "")
    normalized = F.regexp_replace(normalized, r"\s+", " ")
    return normalized


def player_keys(dim_player: DataFrame) -> DataFrame:
    return dim_player.select("player_id", F.explode("all_keys_norm").alias("player_key")).where(
        F.col("player_key").isNotNull()
    )


def enrich_deliveries_players(deliveries: DataFrame, dim_player: DataFrame) -> DataFrame:
    keys = player_keys(dim_player)
    return (
        deliveries.withColumn("striker_key", normalize_player_name(F.col("striker")))
        .withColumn("non_striker_key", normalize_player_name(F.col("non_striker")))
        .withColumn("bowler_key", normalize_player_name(F.col("bowler")))
        .join(
            keys.withColumnRenamed("player_id", "striker_id").withColumnRenamed("player_key", "striker_key"),
            "striker_key",
            "left",
        )
        .join(
            keys.withColumnRenamed("player_id", "non_striker_id").withColumnRenamed(
                "player_key", "non_striker_key"
            ),
            "non_striker_key",
            "left",
        )
        .join(
            keys.withColumnRenamed("player_id", "bowler_id").withColumnRenamed("player_key", "bowler_key"),
            "bowler_key",
            "left",
        )
        .drop("striker_key", "non_striker_key", "bowler_key")
    )


def run(
    context: Optional[PipelineContext] = None,
    spark: Optional[SparkSession] = None,
    sample_only: bool = False,
    write_out: bool = True,
    preview: bool = False,
) -> None:
    context = context or PipelineContext.load()
    spark, should_stop = require_spark("silver-enrich-deliveries-players", spark)
    try:
        dim_player = read_table(spark, context.silver_table_path("dim_player"), context.silver_format)
        deliveries = read_table(spark, context.silver_table_path("deliveries"), context.silver_format)
        if sample_only:
            ids = [r["match_id"] for r in deliveries.select("match_id").distinct().limit(20).collect()]
            deliveries = deliveries.filter(F.col("match_id").isin(ids))

        out = enrich_deliveries_players(deliveries, dim_player)
        if preview:
            show_preview(
                out.select(
                    "match_id",
                    "striker",
                    "striker_id",
                    "non_striker",
                    "non_striker_id",
                    "bowler",
                    "bowler_id",
                ),
                "deliveries with player IDs",
                20,
            )
        if write_out:
            cfg = context.silver["tables"]["deliveries"]
            target = write_table(
                out,
                context.silver_table_path("deliveries"),
                context.silver_format,
                partition_columns=cfg["partition_columns"],
                repartition_columns=["season"],
                atomic=True,
            )
            print(f"Updated silver.deliveries with player IDs at: {target}")
    finally:
        if should_stop:
            spark.stop()


def main(write_out: bool = True, sample_only: bool = False) -> None:
    run(sample_only=sample_only, write_out=write_out and not sample_only, preview=True)


if __name__ == "__main__":
    main(write_out=True, sample_only=False)


