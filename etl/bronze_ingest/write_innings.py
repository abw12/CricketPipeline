from pathlib import Path
from pyspark.sql import functions as F, types as T, DataFrame
from etl.common import get_spark, load_yml, project_root

def write_innings(df:DataFrame, cfg, root_path:Path):
    target_path = cfg["tables"]["innings"]["target_path"]
    parts = cfg["tables"]["innings"]["partition_columns"]
    mode= "overwrite" # first run, later you can change it to some other mode

    out = root_path / target_path
    (df.repartition(1,"season")
        .write.mode(mode)
        .partitionBy(*parts) 
        .format(cfg["storage"]["format"])
        .save(str(out))
    )
    return out

def main(sample_only:bool = True):
    root = project_root()
    cfg = load_yml(str(root / "configs" / "bronze_config.yml"))
    fmt = cfg["storage"]["format"]

    spark = get_spark("bronze-write-innings")

    deliveries_path = root / cfg["tables"]["deliveries"]["target_path"]
    df = spark.read.format(fmt).load(str(deliveries_path))

    # If sampling , keep a small subset of matches for quick iteration
    if sample_only:
        some_matches = [r["match_id"] for r in df.select("match_id").distinct().limit(5).collect()]
        df = df.filter(F.col("match_id").isin(some_matches))
    
    # ---- Legal balls (exclude wides/noballs). Everything else counts as legal. ----
    is_legal_ball = (F.col("extra_wides") == 0) & (F.col("extra_noballs") == 0)

    agg = (
        df.groupBy("match_id","season","inning_no","batting_team")
            .agg(
                F.sum(F.col("runs_total")).cast("int").alias("runs_total"),
                F.sum(F.col("wicket_fell").cast("int")).cast("int").alias("wickets"),
                F.sum(F.when(is_legal_ball,1).otherwise(0)).cast("int").alias("balls_legal")
            )
            .withColumn("overs",F.col("balls_legal") / F.lit(6.0)) # T20: 6 balls per over (legal balls)
    )
    
    # ---- Add lineage for a derived table: clear and explicit ----
    # We can set a synthetic source to indicate it’s rolled up from deliveries

    agg=(
        agg
        .withColumn("src_file_path",F.lit("aggregate:bronze.deliveries"))
        .withColumn("src_file_name",F.lit("aggregate:bronze.deliveries"))
        .withColumn("src_ingest_ts",F.current_timestamp())
        .withColumn(
            "src_record_hash",
            F.sha1(F.concat_ws("|",
                               F.col("match_id"),
                               F.col("inning_no").cast("string"),
                               F.col("runs_total").cast("string"),
                               F.col("wickets").cast("string"),
                               F.col("balls_legal").cast("string"))).cast("string")
        )
    )

    print("\n=== SAMPLE innings rollup (first 10 rows) ===")
    agg.orderBy("match_id","inning_no").show(10,truncate=False)

    print("\n=== innings schema ===")
    agg.printSchema()

    # Write full dataset when ready
    if not sample_only:
        out_dir = write_innings(agg,cfg,root)
        print(f"\n✓ Wrote bronze.innings to: {out_dir}")

        # Read back for sanity
        back = spark.read.format(fmt).load(str(out_dir))
        print("\n=== read-back schema ===")
        back.printSchema()
        print(f"rows: {back.count()}")


    spark.stop()

if __name__ == "__main__":
    main(sample_only=False)