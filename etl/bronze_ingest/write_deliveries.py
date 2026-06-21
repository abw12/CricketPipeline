from __future__ import annotations

from typing import Optional

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, functions as F

from etl.bronze_ingest.deliveries_utils import (
    add_lineage_from_struct,
    compute_delivery_id_expr,
    compute_match_id_from_columns,
)
from etl.common import PipelineContext, require_spark, show_preview, write_table


def season_label(col):
    return F.regexp_replace(col.cast("string"), "/", "-")


def build_deliveries(df_raw: DataFrame) -> DataFrame:
    info = F.col("info")
    teams = info.getField("teams")

    exploded = (
        df_raw.select(
            "src_file_path",
            season_label(info.getField("season")).alias("season"),
            info.getField("venue").alias("venue"),
            F.element_at(info.getField("dates"), 1).cast("string").alias("match_start_date"),
            F.element_at(teams, 1).alias("team1"),
            F.element_at(teams, 2).alias("team2"),
            F.posexplode_outer("innings").alias("inning_pos", "inning"),
        )
        .withColumn("inning_no", (F.col("inning_pos") + F.lit(1)).cast("int"))
        .withColumn("batting_team", F.col("inning.team"))
        .withColumn("overs", F.col("inning.overs"))
        .drop("inning", "inning_pos")
        .select("*", F.posexplode_outer("overs").alias("over_pos", "over_struct"))
        .drop("overs")
        .withColumn(
            "over_no",
            F.coalesce(
                F.col("over_struct.over").cast("int") + F.lit(1),
                (F.col("over_pos") + F.lit(1)).cast("int"),
            ),
        )
        .select("*", F.posexplode_outer(F.col("over_struct.deliveries")).alias("ball_pos", "delivery"))
        .drop("over_struct", "over_pos")
        .withColumn("ball_in_over", (F.col("ball_pos") + F.lit(1)).cast("int"))
        .drop("ball_pos")
        .withColumn("match_id", compute_match_id_from_columns())
        .withColumn(
            "bowling_team",
            F.when(F.col("batting_team") == F.col("team1"), F.col("team2")).otherwise(F.col("team1")),
        )
    )

    d = F.col("delivery")
    w1 = F.element_at(d.getField("wickets"), 1)
    nz = lambda c: F.coalesce(c.cast("int"), F.lit(0))

    shaped = exploded.select(
        "match_id",
        F.col("season").cast("string").alias("season"),
        "inning_no",
        "over_no",
        "ball_in_over",
        "batting_team",
        "bowling_team",
        d.getField("batter").alias("striker"),
        d.getField("non_striker").alias("non_striker"),
        d.getField("bowler").alias("bowler"),
        d.getField("runs").getField("batter").cast("int").alias("runs_batter"),
        d.getField("runs").getField("extras").cast("int").alias("runs_extras"),
        d.getField("runs").getField("total").cast("int").alias("runs_total"),
        nz(d.getField("extras").getField("byes")).alias("extra_byes"),
        nz(d.getField("extras").getField("legbyes")).alias("extra_legbyes"),
        nz(d.getField("extras").getField("wides")).alias("extra_wides"),
        nz(d.getField("extras").getField("noballs")).alias("extra_noballs"),
        F.when(w1.isNotNull(), F.lit(True)).otherwise(F.lit(False)).alias("wicket_fell"),
        w1.getField("player_out").alias("wicket_player_out"),
        w1.getField("kind").alias("wicket_kind"),
        w1.getField("fielders").alias("wicket_fielders"),
        "src_file_path",
        d.alias("delivery_struct_for_hash"),
    ).withColumn("delivery_id", compute_delivery_id_expr())

    return add_lineage_from_struct(shaped, "delivery_struct_for_hash").drop("delivery_struct_for_hash")


def read_raw_deliveries(spark: SparkSession, context: PipelineContext, sample_only: bool) -> DataFrame:
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


def write_deliveries(df: DataFrame, context: PipelineContext) -> Path:
    cfg = context.bronze["tables"]["deliveries"]
    return write_table(
        df,
        target=context.bronze_table_path("deliveries"),
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
    spark, should_stop = require_spark("bronze-write-deliveries", spark)
    try:
        deliveries = build_deliveries(read_raw_deliveries(spark, context, sample_only))
        if preview:
            show_preview(
                deliveries.orderBy("match_id", "inning_no", "over_no", "ball_in_over"),
                "Bronze deliveries",
                20,
            )
        if write_out:
            print(f"Wrote bronze.deliveries to: {write_deliveries(deliveries, context)}")
    finally:
        if should_stop:
            spark.stop()


def main(sample_only: bool = False) -> None:
    run(sample_only=sample_only, write_out=not sample_only, preview=True)


if __name__ == "__main__":
    main(sample_only=False)


