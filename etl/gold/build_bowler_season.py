from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

from etl.common import PipelineContext, read_table, require_spark, show_preview, write_table
from etl.gold.metrics_common import (
    bowling_economy,
    bowling_strike_rate,
    is_bowler_wicket,
    is_legal_for_bowler,
    runs_conceded_col,
    safe_div,
)


def build_bowler_season(deliveries: DataFrame, dim_player: Optional[DataFrame] = None) -> DataFrame:
    base = deliveries.where(F.col("bowler_id").isNotNull()).select(
        "season",
        "match_id",
        "inning_no",
        "over_no",
        "bowler_id",
        is_legal_for_bowler().alias("is_legal"),
        runs_conceded_col().alias("runs_conceded"),
        is_bowler_wicket().alias("is_bowler_wicket"),
    )

    balls = (
        base.withColumn("balls_inc", F.when(F.col("is_legal"), 1).otherwise(0))
        .withColumn("wkts_inc", F.when(F.col("is_bowler_wicket"), 1).otherwise(0))
    )
    over_runs = balls.groupBy("season", "bowler_id", "match_id", "inning_no", "over_no").agg(
        F.sum("runs_conceded").alias("runs_conceded_over"),
        F.sum("balls_inc").alias("balls_legal_over"),
    )
    maidens = over_runs.groupBy("season", "bowler_id").agg(
        F.sum(
            F.when((F.col("runs_conceded_over") == 0) & (F.col("balls_legal_over") > 0), 1).otherwise(0)
        ).alias("maidens")
    )

    out = (
        balls.groupBy("season", "bowler_id")
        .agg(
            F.sum("balls_inc").alias("balls_bowled"),
            F.sum("runs_conceded").alias("runs_conceded"),
            F.sum("wkts_inc").alias("wkts"),
        )
        .join(maidens, ["season", "bowler_id"], "left")
        .fillna({"maidens": 0})
        .withColumn("overs_bowled", (F.col("balls_bowled") / F.lit(6.0)).cast("double"))
        .withColumn(
            "economy",
            bowling_economy(F.col("runs_conceded").cast("double"), F.col("balls_bowled").cast("double")),
        )
        .withColumn(
            "average",
            safe_div(
                F.col("runs_conceded").cast("double"),
                F.when(F.col("wkts") == 0, None).otherwise(F.col("wkts").cast("double")),
            ),
        )
        .withColumn("strike_rate", bowling_strike_rate(F.col("balls_bowled"), F.col("wkts")))
    )

    if dim_player is not None:
        names = dim_player.select("player_id", F.col("name").alias("player_name"))
        out = out.join(names, out["bowler_id"] == names["player_id"], "left").drop("player_id")
    else:
        out = out.withColumn("player_name", F.lit(None).cast("string"))

    return out.select(
        "season",
        "bowler_id",
        "player_name",
        "balls_bowled",
        "overs_bowled",
        "maidens",
        "runs_conceded",
        "wkts",
        "economy",
        "average",
        "strike_rate",
    )


def run(
    context: Optional[PipelineContext] = None,
    spark: Optional[SparkSession] = None,
    sample_only: bool = False,
    write_out: bool = True,
    preview: bool = False,
) -> None:
    context = context or PipelineContext.load()
    spark, should_stop = require_spark("gold-bowler-season", spark)
    try:
        deliveries = read_table(spark, context.silver_table_path("deliveries"), context.silver_format)
        if sample_only:
            ids = [
                r["bowler_id"]
                for r in deliveries.select("bowler_id").where(F.col("bowler_id").isNotNull()).distinct().limit(50).collect()
            ]
            deliveries = deliveries.where(F.col("bowler_id").isin(ids))

        try:
            dim_player = read_table(spark, context.silver_table_path("dim_player"), context.silver_format)
        except Exception:
            dim_player = None

        out = build_bowler_season(deliveries, dim_player)
        if preview:
            show_preview(out.orderBy(F.desc("wkts")), "gold_bowler_season", 20)
        if write_out:
            target = write_table(
                out,
                context.gold_table_path("bowler_season"),
                context.gold_format,
                repartition_columns=["season"],
            )
            print(f"Wrote gold.bowler_season to: {target}")
    finally:
        if should_stop:
            spark.stop()


def main(write_out: bool = True, sample_only: bool = False) -> None:
    run(sample_only=sample_only, write_out=write_out and not sample_only, preview=True)


if __name__ == "__main__":
    main(write_out=True, sample_only=False)


