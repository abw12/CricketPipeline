from __future__ import annotations

from typing import Optional

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, functions as F

from etl.common import PipelineContext, read_table, require_spark, show_preview, write_table


def build_innings(deliveries: DataFrame) -> DataFrame:
    is_legal_ball = (F.col("extra_wides") == 0) & (F.col("extra_noballs") == 0)
    return (
        deliveries.groupBy("match_id", "season", "inning_no", "batting_team")
        .agg(
            F.sum(F.col("runs_total")).cast("int").alias("runs_total"),
            F.sum(F.col("wicket_fell").cast("int")).cast("int").alias("wickets"),
            F.sum(F.when(is_legal_ball, 1).otherwise(0)).cast("int").alias("balls_legal"),
        )
        .withColumn("overs", F.col("balls_legal") / F.lit(6.0))
        .withColumn("src_file_path", F.lit("aggregate:bronze.deliveries"))
        .withColumn("src_file_name", F.lit("aggregate:bronze.deliveries"))
        .withColumn("src_ingest_ts", F.current_timestamp())
        .withColumn(
            "src_record_hash",
            F.sha1(
                F.concat_ws(
                    "|",
                    F.col("match_id"),
                    F.col("inning_no").cast("string"),
                    F.col("runs_total").cast("string"),
                    F.col("wickets").cast("string"),
                    F.col("balls_legal").cast("string"),
                )
            ).cast("string"),
        )
    )


def write_innings(df: DataFrame, context: PipelineContext) -> Path:
    cfg = context.bronze["tables"]["innings"]
    return write_table(
        df,
        target=context.bronze_table_path("innings"),
        fmt=context.bronze_format,
        partition_columns=cfg["partition_columns"],
        repartition_columns=["season"],
    )


def run(
    context: Optional[PipelineContext] = None,
    spark: Optional[SparkSession] = None,
    sample_only: bool = False,
    write_out: bool = True,
    preview: bool = False,
) -> None:
    context = context or PipelineContext.load()
    spark, should_stop = require_spark("bronze-write-innings", spark)
    try:
        deliveries = read_table(spark, context.bronze_table_path("deliveries"), context.bronze_format)
        if sample_only:
            ids = [r["match_id"] for r in deliveries.select("match_id").distinct().limit(5).collect()]
            deliveries = deliveries.filter(F.col("match_id").isin(ids))

        innings = build_innings(deliveries)
        if preview:
            show_preview(innings.orderBy("match_id", "inning_no"), "Bronze innings", 10)
        if write_out:
            print(f"Wrote bronze.innings to: {write_innings(innings, context)}")
    finally:
        if should_stop:
            spark.stop()


def main(sample_only: bool = False) -> None:
    run(sample_only=sample_only, write_out=not sample_only, preview=True)


if __name__ == "__main__":
    main(sample_only=False)


