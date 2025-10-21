from json import load
from pathlib import Path
from numpy import partition
from pyspark.sql import functions as F
from etl.common import get_spark, load_yml, project_root
from etl.silver_transform.register_resolver import make_normalizer, build_player_maps, make_udf_player_resolver

def main(write_out:bool = True, sample_only:bool = True):
    root = project_root()
    cfg_reg = load_yml(str(root / "configs" / "register_config.yml"))
    cfg_silver= load_yml(str(root / "configs" / "silver_config.yml"))
    cfg_bronze= load_yml(str(root / "configs" / "bronze_config.yml"))
    fmt = cfg_silver["storage"]["format"]

    spark = get_spark("silver-enrich-deliveries-players")

    # Load register CSVs

    ppath = root / cfg_reg["paths"]["players"]
    apath = root / cfg_reg["paths"]["player_aliases"]
    colsP = cfg_reg["columns"]["players"]
    colsA = cfg_reg["columns"]["player_aliases"]

    players = spark.read.option("header",True).csv(str(ppath))
    aliases =spark.read.option("header",True).csv(str(apath))

    normalizer = make_normalizer(cfg_reg["name_normalization"])

    # Python dict: (unique_name/name/alias normalized) -> player_id
    player_map = build_player_maps(players, aliases,colsP,colsA,normalizer)
    b_map = spark.sparkContext.broadcast(player_map)
    resolve = make_udf_player_resolver(b_map,normalizer)

    # Read current Silver deliveries (from Step S1C) OR Bronze deliveries if you prefer
    src = root / cfg_silver["tables"]["deliveries"]["target"]
    df = spark.read.format(fmt).load(str(src))

    if sample_only:
        some_matches = [r["match_id"] for r in df.select("match_id").distinct().limit(20).collect()]
        df = df.filter(F.col("match_id").isin(some_matches))
    
    enriched = (df
                .withColumn("striker_id", resolve(F.col("striker")))
                .withColumn("non_striker_id", resolve(F.col("non_striker")))
                .withColumn("bowler_id", resolve(F.col("bowler")))
                )
    print("\n=== deliveries with IDs (preview) ===")
    (enriched
     .select("match_id","season","inning_no","over_no","ball_in_over",
             "striker","striker_id","non_striker","non_striker_id","bowler","bowler_id")
     .orderBy("match_id","inning_no","over_no","ball_in_over")
     .show(20, truncate=False))
    
    if write_out and not sample_only:
        # Overwrite Silver deliveries with the new columns (safe; old columns retained)
        target = root / cfg_silver["tables"]["deliveries"]["target"]
        parts  = cfg_silver["tables"]["deliveries"]["partition_columns"]
        (enriched
         .repartition(1,"season")
         .write.mode("overwrite")
         .partitionBy(*parts)
         .format(fmt)
         .save(str(target))
         )
        print(f"\n✓ Updated Silver deliveries with player IDs at: {target}")
    spark.stop()

if __name__ == "__main__":
    # First run preview; then set sample_only=False to write
    main(write_out=True, sample_only=False)