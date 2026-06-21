from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

from etl.common import PipelineContext, read_table, require_spark, show_preview, write_table
from etl.gold.metrics_common import is_legal_for_bowler


def build_match_summary(matches: DataFrame, deliveries: DataFrame) -> DataFrame:
    innings = deliveries.groupBy("match_id", "inning_no").agg(
        F.sum("runs_total").alias("runs"),
        F.sum(F.col("wicket_fell").cast("int")).alias("wkts"),
        F.sum(F.when(is_legal_for_bowler(), 1).otherwise(0)).alias("balls_legal"),
        F.first("batting_team").alias("batting_team"),
        F.first("batting_team_id").alias("batting_team_id"),
    )

    innings_pivot = innings.groupBy("match_id").pivot("inning_no", [1, 2]).agg(
        F.first("runs").alias("runs"),
        F.first("wkts").alias("wkts"),
        F.first("balls_legal").alias("balls"),
        F.first("batting_team").alias("bat_team"),
        F.first("batting_team_id").alias("bat_team_id"),
    )

    return (
        matches.select(
            "match_id",
            "season",
            "team1",
            "team1_id",
            "team2",
            "team2_id",
            "winner",
            "winner_id",
            "result_std",
            "match_start_dt",
            "venue",
            "city",
            "first_batting_team",
            "first_batting_team_id",
            "second_batting_team",
            "second_batting_team_id",
        )
        .join(innings_pivot, "match_id", "left")
        .withColumn("inning1_runs", F.col("1_runs"))
        .withColumn("inning1_wkts", F.col("1_wkts"))
        .withColumn("inning1_balls", F.col("1_balls"))
        .withColumn("inning2_runs", F.col("2_runs"))
        .withColumn("inning2_wkts", F.col("2_wkts"))
        .withColumn("inning2_balls", F.col("2_balls"))
        .withColumn(
            "chase_success",
            F.when(
                (F.col("winner_id").isNotNull())
                & (F.col("winner_id") == F.col("second_batting_team_id")),
                F.lit(True),
            ).otherwise(F.lit(False)),
        )
        .drop(
            "1_runs",
            "1_wkts",
            "1_balls",
            "1_bat_team",
            "1_bat_team_id",
            "2_runs",
            "2_wkts",
            "2_balls",
            "2_bat_team",
            "2_bat_team_id",
        )
    )


def run(
    context: Optional[PipelineContext] = None,
    spark: Optional[SparkSession] = None,
    sample_only: bool = False,
    write_out: bool = True,
    preview: bool = False,
) -> None:
    context = context or PipelineContext.load()
    spark, should_stop = require_spark("gold-match-summary", spark)
    try:
        matches = read_table(spark, context.silver_table_path("matches"), context.silver_format)
        deliveries = read_table(spark, context.silver_table_path("deliveries"), context.silver_format)
        if sample_only:
            ids = [r["match_id"] for r in matches.select("match_id").limit(50).collect()]
            matches = matches.where(F.col("match_id").isin(ids))
            deliveries = deliveries.where(F.col("match_id").isin(ids))

        out = build_match_summary(matches, deliveries)
        if preview:
            show_preview(out.orderBy("season", "match_start_dt"), "gold_match_summary", 20)
        if write_out:
            target = write_table(
                out,
                context.gold_table_path("match_summary"),
                context.gold_format,
                repartition_columns=["season"],
            )
            print(f"Wrote gold.match_summary to: {target}")
    finally:
        if should_stop:
            spark.stop()


def main(write_out: bool = True, sample_only: bool = False) -> None:
    run(sample_only=sample_only, write_out=write_out and not sample_only, preview=True)


if __name__ == "__main__":
    main(write_out=True, sample_only=False)


