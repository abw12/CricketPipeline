from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

from etl.common import PipelineContext, read_table, require_spark, show_preview, write_table
from etl.silver_transform.normalize_common import normalize_enum, parse_date_multi


def build_batting_order(deliveries: DataFrame) -> DataFrame:
    return (
        deliveries.groupBy("match_id", "inning_no", "batting_team")
        .count()
        .groupBy("match_id")
        .pivot("inning_no")
        .agg(F.first("batting_team"))
        .withColumnRenamed("1", "first_batting_team")
        .withColumnRenamed("2", "second_batting_team")
    )


def normalize_matches(matches: DataFrame, deliveries: Optional[DataFrame], cfg_silver: dict) -> DataFrame:
    toss_map = cfg_silver["enums"]["toss_decision"]["map"]
    result_map = cfg_silver["enums"]["result"]["map"]
    date_patterns = cfg_silver["date_formats"]["match_date"]

    out = (
        matches.withColumn("toss_decision_std", normalize_enum(F.col("toss_decision"), toss_map))
        .withColumn("result_std", normalize_enum(F.col("result"), result_map))
        .withColumn("match_start_dt", parse_date_multi(F.col("match_start_date"), date_patterns))
        .withColumn("match_end_dt", parse_date_multi(F.col("match_end_date"), date_patterns))
    )

    if deliveries is not None:
        out = out.join(build_batting_order(deliveries), "match_id", "left")

    return out.select(
        "match_id",
        "season",
        "match_type",
        "gender",
        "venue",
        "city",
        "match_start_dt",
        "match_end_dt",
        "team1",
        "team2",
        "toss_winner",
        "toss_decision_std",
        "winner",
        "result_std",
        "result_margin",
        "player_of_match",
        "first_batting_team",
        "second_batting_team",
        "src_file_path",
        "src_file_name",
        "src_ingest_ts",
        "src_record_hash",
    )


def run(
    context: Optional[PipelineContext] = None,
    spark: Optional[SparkSession] = None,
    sample_only: bool = False,
    write_out: bool = True,
    preview: bool = False,
) -> None:
    context = context or PipelineContext.load()
    spark, should_stop = require_spark("silver-normalize-matches", spark)
    try:
        matches = read_table(spark, context.bronze_table_path("matches"), context.bronze_format)
        deliveries = None
        if context.silver["enrichment"]["compute_batting_order_from_deliveries"]:
            deliveries = read_table(spark, context.bronze_table_path("deliveries"), context.bronze_format)

        if sample_only:
            ids = [r["match_id"] for r in matches.select("match_id").limit(20).collect()]
            matches = matches.filter(F.col("match_id").isin(ids))
            if deliveries is not None:
                deliveries = deliveries.filter(F.col("match_id").isin(ids))

        out = normalize_matches(matches, deliveries, context.silver)
        if preview:
            show_preview(out.orderBy("season", "match_start_dt"), "Silver matches", 10)
        if write_out:
            cfg = context.silver["tables"]["matches"]
            target = write_table(
                out,
                context.silver_table_path("matches"),
                context.silver_format,
                partition_columns=cfg["partition_columns"],
                repartition_columns=["season"],
            )
            print(f"Wrote silver.matches to: {target}")
    finally:
        if should_stop:
            spark.stop()


def main(write_out: bool = True, sample_only: bool = False) -> None:
    run(sample_only=sample_only, write_out=write_out and not sample_only, preview=True)


if __name__ == "__main__":
    main(write_out=True, sample_only=False)


