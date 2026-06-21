from __future__ import annotations

from typing import Optional

from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from etl.common import PipelineContext, require_spark


def run(
    context: Optional[PipelineContext] = None,
    spark: Optional[SparkSession] = None,
    sample_files: int = 3,
) -> None:
    context = context or PipelineContext.load()
    spark, should_stop = require_spark("bronze-inspect-raw", spark)
    try:
        raw_dir = context.resolve(context.bronze["source"]["raw_path"])
        file_glob = context.bronze["source"].get("file_glob", "*.json")
        multiline = context.bronze["source"].get("read_options", {}).get("multiline", True)
        files = sorted(Path(raw_dir).glob(file_glob))[:sample_files]
        if not files:
            raise FileNotFoundError(f"No JSON files found under {raw_dir} with glob {file_glob}")

        df = (
            spark.read.option("multiLine", str(multiline).lower())
            .json([str(p) for p in files])
            .withColumn("src_file_path", F.input_file_name())
        )

        print("\n=== RAW TOP-LEVEL SCHEMA ===")
        df.printSchema()

        info_cols = [
            "info.season",
            "info.officials",
            "info.match_type",
            "info.gender",
            "info.city",
            "info.venue",
            "info.teams",
            "info.dates",
            "info.toss",
            "info.outcome",
            "info.player_of_match",
            "src_file_path",
        ]
        print("\n=== SAMPLE info columns ===")
        df.select(*[F.col(c) for c in info_cols]).show(sample_files, truncate=False)

        print("\n=== SAMPLE innings size ===")
        df.select(
            F.size("innings").alias("n_innings"),
            F.expr("CASE WHEN size(innings) > 0 THEN innings[0].team ELSE NULL END").alias("inning1_team"),
            "src_file_path",
        ).show(sample_files, truncate=False)
    finally:
        if should_stop:
            spark.stop()


def main() -> None:
    run()


if __name__ == "__main__":
    main()


