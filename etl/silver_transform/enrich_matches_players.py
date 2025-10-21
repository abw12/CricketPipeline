from json import load
from pathlib import Path
from pyspark.sql import functions as F
from etl.common import get_spark, load_yml, project_root
from etl.silver_transform.register_resolver import build_player_maps, make_normalizer, make_udf_player_resolver

def main(write_out:bool=True, sample_only:bool = False):
    root = project_root()
    cfg_reg = load_yml(str(root / "configs"/ "register_config.yml"))
    cfg_silver = load_yml(str(root / "configs" / "silver_config.yml"))
    cfg_bronze= load_yml(str(root / "configs" / "bronze_config.yml"))
    fmt = cfg_silver["storage"]["format"]

    spark = get_spark("enrich-matches-players")

    ppath = cfg_reg["paths"]["players"]
    apath = cfg_reg["paths"]["player_aliases"]
    colsP = cfg_reg["columns"]["players"]
    colsA = cfg_reg["columns"]["player_aliases"]

    players = spark.read.option("header",True).csv(str(ppath))
    aliases = spark.read.option("header",True).csv(str(apath))

    normalizer = make_normalizer(cfg_reg['name_normalization'])
    player_map = build_player_maps(players,aliases,colsP,colsA,normalizer)
    b_map = spark.sparkContext.broadcast(player_map)
    resolve = make_udf_player_resolver(b_map,normalizer)

    # Read Silver matches
    mpath = root / cfg_silver["tables"]["matches"]["target"]
    m = spark.read.format(fmt).load(str(mpath))

    if sample_only:
        some_matches = [r["match_id"] for r in m.select("match_id").distinct().limit(20).collect()]
        m = m.filter(F.col("match_id").isin(some_matches))
    out = m.withColumn("player_of_match_id", resolve(F.col("player_of_match")))

    print("\n=== matches with player_of_match_id (preview) ===")
    out.select("match_id","season","player_of_match","player_of_match_id").orderBy("season").show(20, truncate=False)

    if write_out and not sample_only:
        target = root / cfg_silver["tables"]["matches"]["target"]
        parts  = cfg_silver["tables"]["matches"]["partition_columns"]
        (out
         .repartition(1, "season")
         .write.mode("overwrite")
         .partitionBy(*parts)
         .format(fmt)
         .save(str(target)))
        print(f"\n✓ Updated Silver matches with player_of_match_id at: {target}")

    spark.stop()

if __name__ == "__main__":
    main(write_out=False, sample_only=True)