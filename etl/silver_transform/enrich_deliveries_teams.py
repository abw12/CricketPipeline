from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

from etl.common import PipelineContext, read_table, require_spark, show_preview, write_table
from etl.silver_transform.build_dim_team import normalize_team_col


def team_keys(dim_team: DataFrame) -> DataFrame:
    return dim_team.select(
        "team_id",
        F.explode(F.array_union(F.array(F.col("name_canonical_norm")), F.col("aliases_norm"))).alias("key_norm"),
    )


def enrich_deliveries_teams(deliveries: DataFrame, dim_team: DataFrame, cfg_silver: dict) -> DataFrame:
    keys = team_keys(dim_team)

    joined = (
        deliveries.withColumn("bat_key", normalize_team_col(F.col("batting_team")))
        .withColumn("bowl_key", normalize_team_col(F.col("bowling_team")))
        .join(
            keys.withColumnRenamed("team_id", "batting_team_id").withColumnRenamed("key_norm", "bat_key"),
            "bat_key",
            "left",
        )
        .join(
            keys.withColumnRenamed("team_id", "bowling_team_id").withColumnRenamed("key_norm", "bowl_key"),
            "bowl_key",
            "left",
        )
    )
    return joined.drop("bat_key", "bowl_key")


def run(
    context: Optional[PipelineContext] = None,
    spark: Optional[SparkSession] = None,
    sample_only: bool = False,
    write_out: bool = True,
    preview: bool = False,
) -> None:
    context = context or PipelineContext.load()
    spark, should_stop = require_spark("silver-enrich-deliveries-teams", spark)
    try:
        dim_team = read_table(spark, context.silver_table_path("dim_team"), context.silver_format)
        deliveries = read_table(spark, context.silver_table_path("deliveries"), context.silver_format)
        if sample_only:
            ids = [r["match_id"] for r in deliveries.select("match_id").distinct().limit(20).collect()]
            deliveries = deliveries.filter(F.col("match_id").isin(ids))

        out = enrich_deliveries_teams(deliveries, dim_team, context.silver)
        if preview:
            show_preview(
                out.select(
                    "match_id",
                    "batting_team",
                    "batting_team_id",
                    "bowling_team",
                    "bowling_team_id",
                ),
                "deliveries with team IDs",
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
            print(f"Updated silver.deliveries with team IDs at: {target}")
    finally:
        if should_stop:
            spark.stop()


def main(write_out: bool = True, sample_only: bool = False) -> None:
    run(sample_only=sample_only, write_out=write_out and not sample_only, preview=True)


if __name__ == "__main__":
    main(write_out=True, sample_only=False)


