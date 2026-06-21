from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

from etl.common import PipelineContext, read_table, require_spark, show_preview, write_table
from etl.gold.metrics_common import batting_strike_rate, is_legal_for_batter, safe_div


def build_batter_season(deliveries: DataFrame, dim_player: Optional[DataFrame] = None) -> DataFrame:
    appearances = (
        deliveries.where(F.col("striker_id").isNotNull())
        .select("season", "match_id", "inning_no", "striker_id")
        .distinct()
    )
    innings_batted = appearances.groupBy("season", "striker_id").agg(
        F.countDistinct("match_id", "inning_no").alias("innings_batted")
    )
    matches_played = appearances.groupBy("season", "striker_id").agg(
        F.countDistinct("match_id").alias("matches_played")
    )

    balls = deliveries.where(F.col("striker_id").isNotNull()).select(
        "season",
        "striker_id",
        F.col("runs_batter").cast("int").alias("runs_batter"),
        F.when(is_legal_for_batter(), 1).otherwise(0).alias("bf_inc"),
        F.when(F.col("runs_batter") == 4, 1).otherwise(0).alias("fours_inc"),
        F.when(F.col("runs_batter") == 6, 1).otherwise(0).alias("sixes_inc"),
        F.when(
            (F.col("wicket_fell") == True) & (F.col("wicket_player_out") == F.col("striker")),
            1,
        )
        .otherwise(0)
        .alias("outs_inc"),
    )

    out = (
        balls.groupBy("season", "striker_id")
        .agg(
            F.sum("runs_batter").alias("runs"),
            F.sum("bf_inc").alias("balls_faced"),
            F.sum("fours_inc").alias("fours"),
            F.sum("sixes_inc").alias("sixes"),
            F.sum("outs_inc").alias("outs"),
        )
        .join(innings_batted, ["season", "striker_id"], "left")
        .join(matches_played, ["season", "striker_id"], "left")
        .withColumn("average", safe_div(F.col("runs").cast("double"), F.col("outs").cast("double")))
        .withColumn(
            "strike_rate",
            batting_strike_rate(F.col("runs").cast("double"), F.col("balls_faced").cast("double")),
        )
    )

    if dim_player is not None:
        names = dim_player.select("player_id", F.col("name").alias("player_name"))
        out = out.join(names, out["striker_id"] == names["player_id"], "left").drop("player_id")
    else:
        out = out.withColumn("player_name", F.lit(None).cast("string"))

    return out.select(
        "season",
        "striker_id",
        "player_name",
        "matches_played",
        "innings_batted",
        "runs",
        "balls_faced",
        "fours",
        "sixes",
        "outs",
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
    spark, should_stop = require_spark("gold-batter-season", spark)
    try:
        deliveries = read_table(spark, context.silver_table_path("deliveries"), context.silver_format)
        if sample_only:
            ids = [
                r["striker_id"]
                for r in deliveries.select("striker_id").where(F.col("striker_id").isNotNull()).distinct().limit(50).collect()
            ]
            deliveries = deliveries.where(F.col("striker_id").isin(ids))

        try:
            dim_player = read_table(spark, context.silver_table_path("dim_player"), context.silver_format)
        except Exception:
            dim_player = None

        out = build_batter_season(deliveries, dim_player)
        if preview:
            show_preview(out.orderBy(F.desc("runs")), "gold_batter_season", 20)
        if write_out:
            target = write_table(
                out,
                context.gold_table_path("batter_season"),
                context.gold_format,
                repartition_columns=["season"],
            )
            print(f"Wrote gold.batter_season to: {target}")
    finally:
        if should_stop:
            spark.stop()


def main(write_out: bool = True, sample_only: bool = False) -> None:
    run(sample_only=sample_only, write_out=write_out and not sample_only, preview=True)


if __name__ == "__main__":
    main(write_out=True, sample_only=False)


