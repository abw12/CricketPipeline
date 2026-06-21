from __future__ import annotations

from typing import Optional

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, functions as F

from etl.bronze_ingest.matches_utils import add_lineage, compute_match_id_expr
from etl.common import (
    PipelineContext,
    require_spark,
    show_preview,
    write_table,
)


def season_label(col):
    return F.regexp_replace(col.cast("string"), "/", "-")


def select_match_fields(df_raw: DataFrame) -> DataFrame:
    info = F.col("info")
    outcome = info.getField("outcome")
    teams = info.getField("teams")
    toss = info.getField("toss")

    df = df_raw.select(
        info.alias("info_struct"),
        F.col("src_file_path"),
        season_label(info.getField("season")).alias("season"),
        info.getField("match_type").alias("match_type"),
        info.getField("gender").alias("gender"),
        info.getField("venue").alias("venue"),
        info.getField("city").alias("city"),
        F.element_at(info.getField("dates"), 1).cast("string").alias("match_start_date"),
        F.element_at(info.getField("dates"), 2).cast("string").alias("match_end_date"),
        F.element_at(teams, 1).alias("team1"),
        F.element_at(teams, 2).alias("team2"),
        toss.getField("winner").alias("toss_winner"),
        toss.getField("decision").alias("toss_decision"),
        outcome.getField("winner").alias("winner"),
        F.when(outcome.getField("by").getField("runs").isNotNull(), F.lit("runs"))
        .when(outcome.getField("by").getField("wickets").isNotNull(), F.lit("wickets"))
        .otherwise(F.lit(None))
        .alias("result"),
        F.coalesce(
            outcome.getField("by").getField("runs"),
            outcome.getField("by").getField("wickets"),
        )
        .cast("int")
        .alias("result_margin"),
        F.element_at(info.getField("player_of_match"), 1).alias("player_of_match"),
        info.getField("officials").alias("officials"),
    )

    return df.withColumn("match_id", compute_match_id_expr()).select(
        "match_id",
        "season",
        "match_type",
        "gender",
        "venue",
        "city",
        "match_start_date",
        "match_end_date",
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "winner",
        "result",
        "result_margin",
        "player_of_match",
        "officials",
        "info_struct",
        "src_file_path",
    )


def build_matches(df_raw: DataFrame) -> DataFrame:
    return add_lineage(select_match_fields(df_raw)).drop("info_struct")


def read_raw_matches(spark: SparkSession, context: PipelineContext, sample_only: bool) -> DataFrame:
    raw_dir = context.resolve(context.bronze["source"]["raw_path"])
    file_glob = context.bronze["source"].get("file_glob", "*.json")
    multiline = context.bronze["source"].get("read_options", {}).get("multiline", True)
    files = sorted(Path(raw_dir).glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No JSON files under {raw_dir} with glob {file_glob}")

    paths = [str(p) for p in (files[:50] if sample_only else files)]
    return (
        spark.read.option("multiLine", str(multiline).lower())
        .json(paths)
        .withColumn("src_file_path", F.input_file_name())
    )


def write_matches(df: DataFrame, context: PipelineContext) -> Path:
    cfg = context.bronze["tables"]["matches"]
    return write_table(
        df,
        target=context.bronze_table_path("matches"),
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
    spark, should_stop = require_spark("bronze-write-matches", spark)
    try:
        matches = build_matches(read_raw_matches(spark, context, sample_only))
        if preview:
            show_preview(matches.orderBy("season", "match_start_date"), "Bronze matches", 5)
        if write_out:
            print(f"Wrote bronze.matches to: {write_matches(matches, context)}")
    finally:
        if should_stop:
            spark.stop()


def main(sample_only: bool = False) -> None:
    run(sample_only=sample_only, write_out=not sample_only, preview=True)


if __name__ == "__main__":
    main(sample_only=False)


