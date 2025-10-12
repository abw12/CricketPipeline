from pyspark.sql import functions as F, types as T
from pathlib import Path
from etl.common import get_spark, load_yml, project_root

def main(write_out:bool = True, sample_only:bool = True):
    root = project_root()
    cfg_silver = load_yml(str( root / "configs" / "silver_config.yml"))
    cfg_bronze = load_yml(str( root / "configs" / "bronze_config.yml"))
    fmt = cfg_silver["storage"]["format"]

    spark = get_spark("silver-normalize-deliveries")

    bronze_deliveries_path = root / cfg_bronze["tables"]["deliveries"]["target_path"]
    df = spark.read.format(fmt).load(str(bronze_deliveries_path))

    if sample_only:
        some_matches = [r["match_id"] for r in df.select("match_id").distinct().limit(10).collect()]
        df = df.filter(F.col("match_id").isin(some_matches))

    # Enforce types (some columns may be null-typed if empty in a partition)
    casted = (df
        .withColumn("season",        F.col("season").cast(T.IntegerType()))
        .withColumn("inning_no",     F.col("inning_no").cast(T.IntegerType()))
        .withColumn("over_no",       F.col("over_no").cast(T.IntegerType()))
        .withColumn("ball_in_over",  F.col("ball_in_over").cast(T.IntegerType()))
        .withColumn("runs_batter",   F.col("runs_batter").cast(T.IntegerType()))
        .withColumn("runs_extras",   F.col("runs_extras").cast(T.IntegerType()))
        .withColumn("runs_total",    F.col("runs_total").cast(T.IntegerType()))
        .withColumn("extra_byes",    F.col("extra_byes").cast(T.IntegerType()))
        .withColumn("extra_legbyes", F.col("extra_legbyes").cast(T.IntegerType()))
        .withColumn("extra_wides",   F.col("extra_wides").cast(T.IntegerType()))
        .withColumn("extra_noballs", F.col("extra_noballs").cast(T.IntegerType()))
        .withColumn("wicket_fell",   F.col("wicket_fell").cast(T.BooleanType()))
    )

    # Column ordering for Silver deliveries
    out = casted.select(
        "delivery_id", "match_id", "season",
        "inning_no", "over_no", "ball_in_over",
        "batting_team", "bowling_team",
        "striker", "non_striker", "bowler",
        "runs_batter", "runs_extras", "runs_total",
        "extra_byes", "extra_legbyes", "extra_wides", "extra_noballs",
        "wicket_fell", "wicket_player_out", "wicket_kind", "wicket_fielders",
        "src_file_path", "src_file_name", "src_ingest_ts", "src_record_hash",
    )

    print("\n=== SAMPLE Silver deliveries (first 20) ===")
    out.orderBy("match_id", "inning_no", "over_no", "ball_in_over").show(20, truncate=False)
    out.printSchema()

    if write_out and not sample_only:
        target = root / cfg_silver["tables"]["deliveries"]["target"]
        parts = cfg_silver["tables"]["deliveries"]["partition_columns"]
        (out
            .repartition(1,"season")
            .write.mode("overwrite")
            .partitionBy(*parts)
            .format(fmt)
            .save(str(target)))
        print(f"\n✓ Wrote Silver deliveries to: {target}")
    spark.stop()

if __name__ == "__main__":
    main(write_out=True,sample_only=False)