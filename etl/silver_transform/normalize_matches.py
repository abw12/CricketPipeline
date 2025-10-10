from pathlib import Path
from pyspark.sql import functions as F
from etl.common import get_spark, load_yml, project_root
from etl.silver_transform.normalize_common import normalize_enum, parse_date_multi

def main(write_out:bool = True, sample_only:bool=True):

    root = project_root()
    cfg_silver = load_yml(str(root / "configs" / "silver_config.yml"))
    cfg_bronze = load_yml(str(root / "configs" / "bronze_config.yml"))
    fmt = cfg_silver["storage"]["format"]

    spark = get_spark("silver-normalize-matches")

    bronze_matches_path = root / cfg_bronze["tables"]["matches"]["target_path"]
    matches_bz = spark.read.format(fmt).load(str(bronze_matches_path))

    if cfg_silver["enrichment"]["compute_batting_order_from_deliveries"]:
        bronze_deliveries_path = root / cfg_bronze["tables"]["deliveries"]["target_path"]
        deliveries_bz = spark.read.format(fmt).load(str(bronze_deliveries_path))

        if sample_only:
            #spped up first preview
            some_matches = [r["match_id"] for r in matches_bz.select("match_id").limit(20).collect()]
            deliveries_bz = deliveries_bz.filter(F.col("match_id").isin(some_matches))
            matches_bz = matches_bz.filter(F.col("match_id").isin(some_matches))

        bat_order = (deliveries_bz
                     .groupBy("match_id","inning_no","batting_team")
                     .count()
                     .groupBy("match_id")
                     .pivot("inning_no")
                     .agg(F.first("batting_team"))
                     .withColumnRenamed("1","first_batting_team")
                     .withColumnRenamed("2","second_batting_team"))
    else:
        bat_order = None
    
    # --- Normalize enums and dates ---
    toss_map = cfg_silver["enums"]["toss_decision"]["map"]
    result_map = cfg_silver["enums"]["result"]["map"]
    date_pats  = cfg_silver["date_formats"]["match_date"]

    m = (matches_bz
        .withColumn("toss_decision_std", normalize_enum(F.col("toss_decision"),toss_map))
        .withColumn("result_std", normalize_enum(F.col("result"),result_map))
        .withColumn("match_start_dt", parse_date_multi(F.col("match_start_date"),date_pats))
        .withColumn("match_end_dt", parse_date_multi(F.col("match_end_date"),date_pats))
        )

    if bat_order is not None:
        m = m.join(bat_order, "match_id", "left")
    
    # --- Column ordering for Silver ---
    out = m.select(
        # IDs & partitions
        "match_id", "season",
        "match_type", "gender",
        "venue", "city",

        # Dates normalized
        "match_start_dt", "match_end_dt",

        # Teams & toss/result (normalized)
        "team1", "team2",
        "toss_winner", "toss_decision_std",
        "winner", "result_std", "result_margin", "player_of_match",

        # Enrichment
        "first_batting_team", "second_batting_team",

        # Lineage (copied from Bronze)
        "src_file_path", "src_file_name", "src_ingest_ts", "src_record_hash",
    )

    print("\n=== SAMPLE Silver matches (first 10) ===")
    out.orderBy("season", "match_start_dt").show(10, truncate=False)
    out.printSchema()

    if write_out and not sample_only:
        target = root / cfg_silver["tables"]["matches"]["target"]
        parts  = cfg_silver["tables"]["matches"]["partition_columns"]

        (out
         .repartition(1, "season")
         .write.mode("overwrite")
         .partitionBy(*parts)
         .format(fmt)
         .save(str(target)))

        print(f"\n✓ Wrote Silver matches to: {target}")

    spark.stop()

if __name__ == "__main__":
    main(write_out=True, sample_only=False)  # preview first

