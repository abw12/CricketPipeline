from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

from etl.common import PipelineContext, read_table, require_spark, show_preview, write_table


def normalize_deliveries(deliveries: DataFrame) -> DataFrame:
    casted = (
        deliveries.withColumn("season", F.col("season").cast(T.StringType()))
        .withColumn("inning_no", F.col("inning_no").cast(T.IntegerType()))
        .withColumn("over_no", F.col("over_no").cast(T.IntegerType()))
        .withColumn("ball_in_over", F.col("ball_in_over").cast(T.IntegerType()))
        .withColumn("runs_batter", F.col("runs_batter").cast(T.IntegerType()))
        .withColumn("runs_extras", F.col("runs_extras").cast(T.IntegerType()))
        .withColumn("runs_total", F.col("runs_total").cast(T.IntegerType()))
        .withColumn("extra_byes", F.col("extra_byes").cast(T.IntegerType()))
        .withColumn("extra_legbyes", F.col("extra_legbyes").cast(T.IntegerType()))
        .withColumn("extra_wides", F.col("extra_wides").cast(T.IntegerType()))
        .withColumn("extra_noballs", F.col("extra_noballs").cast(T.IntegerType()))
        .withColumn("wicket_fell", F.col("wicket_fell").cast(T.BooleanType()))
    )

    return casted.select(
        "delivery_id",
        "match_id",
        "season",
        "inning_no",
        "over_no",
        "ball_in_over",
        "batting_team",
        "bowling_team",
        "striker",
        "non_striker",
        "bowler",
        "runs_batter",
        "runs_extras",
        "runs_total",
        "extra_byes",
        "extra_legbyes",
        "extra_wides",
        "extra_noballs",
        "wicket_fell",
        "wicket_player_out",
        "wicket_kind",
        "wicket_fielders",
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
    spark, should_stop = require_spark("silver-normalize-deliveries", spark)
    try:
        deliveries = read_table(spark, context.bronze_table_path("deliveries"), context.bronze_format)
        if sample_only:
            ids = [r["match_id"] for r in deliveries.select("match_id").distinct().limit(10).collect()]
            deliveries = deliveries.filter(F.col("match_id").isin(ids))

        out = normalize_deliveries(deliveries)
        if preview:
            show_preview(
                out.orderBy("match_id", "inning_no", "over_no", "ball_in_over"),
                "Silver deliveries",
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
            )
            print(f"Wrote silver.deliveries to: {target}")
    finally:
        if should_stop:
            spark.stop()


def main(write_out: bool = True, sample_only: bool = False) -> None:
    run(sample_only=sample_only, write_out=write_out and not sample_only, preview=True)


if __name__ == "__main__":
    main(write_out=True, sample_only=False)


