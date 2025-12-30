# etl/gold/build_batter_season.py
from pyspark.sql import functions as F
import os
import sys

# Add parent directory to Python path to allow module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from etl.common import get_spark, load_yml, project_root
from etl.gold.metrics_common import (
    is_legal_for_batter, batting_strike_rate, safe_div
)

def main(write_out: bool = True, sample_only: bool = True):
    root = project_root()
    cfg_silver = load_yml(str(root / "configs" / "silver_config.yml"))
    fmt = cfg_silver["storage"]["format"]

    spark = get_spark("gold-batter-season")

    #Read silver deliveries ( must include striker_id etc.)
    d_path = root / cfg_silver["tables"]["deliveries"]["target"]
    d = spark.read.format(fmt).load(str(d_path))

    if sample_only:
        some_players = [r["striker_id"] for r in d.select("striker_id").where(F.col("striker_id").isNotNull()).distinct().limit(50).collect()]
        d = d.where(F.col("striker_id").isin(some_players))
    
    # Appearances by batter
    appears = d.where(F.col("striker_id").isNotNull()) \
                .select("season","match_id","inning_no","striker_id") \
                .distinct()
    
    innings_batted = appears.groupBy("season","striker_id").agg(F.countDistinct("inning_no","match_id").alias("innings_batted"))
    matches_played = appears.groupBy("season", "striker_id").agg(F.countDistinct("match_id").alias("matches_played"))

    # Per-ball contributions
    batter_balls = d.where(F.col("striker_id").isNotNull()) \
            .select(
            "season", "striker_id",
            F.col("runs_batter").cast("int").alias("runs_batter"),
            F.when(is_legal_for_batter(), 1).otherwise(0).alias("bf_inc"),
            F.when(F.col("runs_batter") == 4, 1).otherwise(0).alias("fours_inc"),
            F.when(F.col("runs_batter") == 6, 1).otherwise(0).alias("sixes_inc"),
            # Out when the wicket fell and the player_out is the striker
            F.when((F.col("wicket_fell") == True) & (F.col("wicket_player_out") == F.col("striker")), 1).otherwise(0).alias("outs_inc"),
        )
    
    agg = (batter_balls
         .groupBy("season","striker_id")
         .agg(
             F.sum("runs_batter").alias("runs"),
             F.sum("bf_inc").alias("balls_faced"),
             F.sum("fours_inc").alias("fours"),
             F.sum("sixes_inc").alias("sixes"),
             F.sum("outs_inc").alias("outs")
        ))
    
    # Join appearances
    gold = (agg
        .join(innings_batted, ["season","striker_id"], "left")
        .join(matches_played, ["season","striker_id"], "left")
        .withColumn("average", safe_div(F.col("runs").cast("double"), F.col("outs").cast("double")))
        .withColumn("strike_rate", batting_strike_rate(F.col("runs").cast("double"), F.col("balls_faced").cast("double")))
    )

    # Optional: bring readable name from dim_player (if you built it)
    dim_player_path = root / cfg_silver["tables"]["dim_player"]["target"]
    try:
        dim = spark.read.format(fmt).load(str(dim_player_path)).select("player_id", F.col("name").alias("player_name"))
        gold = gold.join(dim, gold["striker_id"] == dim["player_id"], "left").drop("player_id")
    except Exception:
        pass  # if dim_player not present, skip

        # Arrange columns
    out = gold.select(
        "season", "striker_id", "player_name",
        "matches_played", "innings_batted",
        "runs", "balls_faced", "fours", "sixes", "outs",
        "average", "strike_rate"
    )

    print("\n=== gold_batter_season sample ===")
    out.orderBy(F.desc("runs")).show(20, truncate=False)
    out.printSchema()

    if write_out and not sample_only:
        target = root / cfg_silver["gold"]["tables"]["batter_season"]
        (out
         .repartition(1, "season")
         .write.mode("overwrite")
         .format(cfg_silver["gold"]["format"])
         .save(str(target))
        )
        print(f"\n✓ Wrote gold_batter_season to: {target}") 

    spark.stop()

if __name__ == "__main__":
    # preview first; then set sample_only=False to write
    main(write_out=True, sample_only=False)