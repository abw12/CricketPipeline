from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

from etl.common import PipelineContext, read_table, require_spark, show_preview, write_table
from etl.gold.metrics_common import (
    is_bowler_wicket,
    is_legal_for_batter,
    is_legal_for_bowler,
    runs_conceded_col,
    safe_div,
)
from etl.gold.phase_common import phase_col


def build_team_phase_season(deliveries: DataFrame, dim_team: Optional[DataFrame] = None) -> DataFrame:
    d = (
        deliveries.withColumn("phase", phase_col(F.col("over_no")))
        .withColumn("bat_legal", F.when(is_legal_for_batter(), 1).otherwise(0))
        .withColumn("bowl_legal", F.when(is_legal_for_bowler(), 1).otherwise(0))
        .withColumn("runs_conceded", runs_conceded_col())
        .withColumn("bowler_wkt", F.when(is_bowler_wicket(), 1).otherwise(0))
        .withColumn("wkt_lost", F.when(F.col("wicket_fell") == True, 1).otherwise(0))
        .withColumn("four", F.when(F.col("runs_batter") == 4, 1).otherwise(0))
        .withColumn("six", F.when(F.col("runs_batter") == 6, 1).otherwise(0))
    )

    batting = (
        d.where(F.col("batting_team_id").isNotNull())
        .groupBy("season", "batting_team_id", "phase")
        .agg(
            F.sum("runs_total").alias("runs_scored"),
            F.sum("bat_legal").alias("balls_faced"),
            F.sum("four").alias("fours"),
            F.sum("six").alias("sixes"),
            F.sum("wkt_lost").alias("wickets_lost"),
        )
        .withColumn("overs_faced", F.col("balls_faced") / F.lit(6.0))
        .withColumn("run_rate", safe_div(F.col("runs_scored") * F.lit(6.0), F.col("balls_faced")))
        .withColumnRenamed("batting_team_id", "team_id")
    )

    bowling = (
        d.where(F.col("bowling_team_id").isNotNull())
        .groupBy("season", "bowling_team_id", "phase")
        .agg(
            F.sum("runs_conceded").alias("runs_conceded"),
            F.sum("bowl_legal").alias("balls_bowled"),
            F.sum("bowler_wkt").alias("wkts_taken"),
        )
        .withColumn("overs_bowled", F.col("balls_bowled") / F.lit(6.0))
        .withColumn("economy", safe_div(F.col("runs_conceded") * F.lit(6.0), F.col("balls_bowled")))
        .withColumnRenamed("bowling_team_id", "team_id")
    )

    out = (
        batting.join(bowling, ["season", "team_id", "phase"], "full")
        .fillna(
            0,
            subset=[
                "runs_scored",
                "balls_faced",
                "fours",
                "sixes",
                "wickets_lost",
                "runs_conceded",
                "balls_bowled",
                "wkts_taken",
            ],
        )
        .withColumn("run_rate", F.when(F.col("balls_faced") == 0, None).otherwise(F.col("run_rate")))
        .withColumn("economy", F.when(F.col("balls_bowled") == 0, None).otherwise(F.col("economy")))
    )

    if dim_team is not None:
        names = dim_team.select("team_id", F.col("name_canonical").alias("team_name"))
        out = out.join(names, "team_id", "left")
    else:
        out = out.withColumn("team_name", F.lit(None).cast("string"))

    return out.select(
        "season",
        "team_id",
        "team_name",
        "phase",
        "runs_scored",
        "balls_faced",
        "overs_faced",
        "run_rate",
        "fours",
        "sixes",
        "wickets_lost",
        "runs_conceded",
        "balls_bowled",
        "overs_bowled",
        "economy",
        "wkts_taken",
    )


def run(
    context: Optional[PipelineContext] = None,
    spark: Optional[SparkSession] = None,
    sample_only: bool = False,
    write_out: bool = True,
    preview: bool = False,
) -> None:
    context = context or PipelineContext.load()
    spark, should_stop = require_spark("gold-team-phase-season", spark)
    try:
        deliveries = read_table(spark, context.silver_table_path("deliveries"), context.silver_format)
        if sample_only:
            seasons = [r["season"] for r in deliveries.select("season").distinct().orderBy("season").limit(2).collect()]
            deliveries = deliveries.where(F.col("season").isin(seasons))

        try:
            dim_team = read_table(spark, context.silver_table_path("dim_team"), context.silver_format)
        except Exception:
            dim_team = None

        out = build_team_phase_season(deliveries, dim_team)
        if preview:
            show_preview(out.orderBy("season", "team_name", "phase"), "gold_team_phase_season", 30)
        if write_out:
            target = write_table(
                out,
                context.gold_table_path("team_phase_season"),
                context.gold_format,
                repartition_columns=["season"],
            )
            print(f"Wrote gold.team_phase_season to: {target}")
    finally:
        if should_stop:
            spark.stop()


def main(write_out: bool = True, sample_only: bool = False) -> None:
    run(sample_only=sample_only, write_out=write_out and not sample_only, preview=True)


if __name__ == "__main__":
    main(write_out=True, sample_only=False)


