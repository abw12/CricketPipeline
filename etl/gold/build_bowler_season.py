# etl/gold/build_bowler_season.py
from pathlib import Path
from pyspark.sql import functions as F, types as T, Window
from etl.common import get_spark, load_yml, project_root
from etl.gold.metrics_common import (
    is_legal_for_bowler, runs_conceded_col, is_bowler_wicket,
    bowling_economy, bowling_strike_rate, safe_div
)

def main(write_out: bool = True, sample_only: bool = True):
    root = project_root()
    cfg_silver = load_yml(str(root / "configs" / "silver_config.yml"))
    fmt = cfg_silver["storage"]["format"]

    spark = get_spark("gold-bowler-season")

    d_path = root / cfg_silver["tables"]["deliveries"]["target"]
    d = spark.read.format(fmt).load(str(d_path))

    if sample_only:
        some_bowlers = [r["bowler_id"] for r in d.select("bowler_id").where(F.col("bowler_id").isNotNull()).distinct().limit(50).collect()]
        d = d.where(F.col("bowler_id").isin(some_bowlers))

    # Base columns for bowling metrics
    base = (d.where(F.col("bowler_id").isNotNull())
              .select(
                  "season", "match_id", "inning_no", "over_no", "ball_in_over", "bowler_id",
                  is_legal_for_bowler().alias("is_legal"),
                  runs_conceded_col().alias("runs_conceded"),
                  is_bowler_wicket().alias("is_bowler_wicket")
              ))

    # Balls and runs per ball
    bowls = (base
        .withColumn("balls_inc", F.when(F.col("is_legal"), 1).otherwise(0))
        .withColumn("wkts_inc",  F.when(F.col("is_bowler_wicket"), 1).otherwise(0))
    )

    # Maidens require over-level view
    w_over = Window.partitionBy("season", "bowler_id", "match_id", "inning_no", "over_no")
    over_runs = bowls.groupBy("season","bowler_id","match_id","inning_no","over_no") \
                     .agg(F.sum("runs_conceded").alias("runs_conceded_over"),
                          F.sum("balls_inc").alias("balls_legal_over"))
    maidens_df = over_runs.select(
        "season","bowler_id","match_id","inning_no","over_no",
        F.when((F.col("runs_conceded_over") == 0) & (F.col("balls_legal_over") > 0), 1).otherwise(0).alias("is_maiden_over")
    )

    # Season aggregation
    agg = (bowls.groupBy("season", "bowler_id")
        .agg(
            F.sum("balls_inc").alias("balls_bowled"),
            F.sum("runs_conceded").alias("runs_conceded"),
            F.sum("wkts_inc").alias("wkts"),
        ))

    maidens = (maidens_df.groupBy("season","bowler_id")
               .agg(F.sum("is_maiden_over").alias("maidens")))

    gold = (agg.join(maidens, ["season","bowler_id"], "left")
        .fillna({"maidens": 0})
        .withColumn("overs_bowled", (F.col("balls_bowled") / F.lit(6.0)).cast("double"))
        .withColumn("economy",      bowling_economy(F.col("runs_conceded").cast("double"), F.col("balls_bowled").cast("double")))
        .withColumn("average",      safe_div(F.col("runs_conceded").cast("double"), F.when(F.col("wkts") == 0, None).otherwise(F.col("wkts").cast("double"))))
        .withColumn("strike_rate",  bowling_strike_rate(F.col("balls_bowled"), F.col("wkts")))
    )

    # Optional name
    dim_player_path = root / cfg_silver["tables"]["dim_player"]["target"]
    try:
        dim = spark.read.format(fmt).load(str(dim_player_path)).select("player_id", F.col("name").alias("player_name"))
        gold = gold.join(dim, gold["bowler_id"] == dim["player_id"], "left").drop("player_id")
    except Exception:
        pass

    out = gold.select(
        "season", "bowler_id", "player_name",
        "balls_bowled", "overs_bowled", "maidens",
        "runs_conceded", "wkts",
        "economy", "average", "strike_rate"
    )

    print("\n=== gold_bowler_season sample ===")
    out.orderBy(F.desc("wkts")).show(20, truncate=False)
    out.printSchema()

    if write_out and not sample_only:
        target = root / cfg_silver["gold"]["tables"]["bowler_season"]
        (out
         .repartition(1, "season")
         .write.mode("overwrite")
         .format(cfg_silver["gold"]["format"])
         .save(str(target)))
        print(f"\n✓ Wrote gold_bowler_season to: {target}")

    spark.stop()

if __name__ == "__main__":
    # preview first; then set sample_only=False to write
    main(write_out=True, sample_only=False)
