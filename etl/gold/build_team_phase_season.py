# etl/gold/build_team_phase_season.py
from ast import alias
from pathlib import Path
from pyspark.sql import functions as F

from etl.common import get_spark, load_yml, project_root
from etl.gold.phase_common import phase_col
from etl.gold.metrics_common import (
    is_legal_for_batter, is_legal_for_bowler,
    runs_conceded_col, is_bowler_wicket, safe_div
)

def main(write_out:bool = True,sample_only:bool = True):
    root = project_root()
    cfg = load_yml(str(root / "configs" / "silver_config.yml"))
    fmt = cfg["storage"]["format"]

    spark = get_spark("gold-team-phase-season")

    # Read enriched Silver deliveries (must include team_id columns)
    d_path = root / cfg["tables"]["deliveries"]["target"]
    d = spark.read.format(fmt).load(str(d_path))

    if sample_only:
        seasons = [r["season"] for r in d.select("season").distinct().orderBy("season").limit(2).collect()]
        d = d.where(F.col("season").isin(seasons))
    
    d2 = (d
        .withColumn("phase", phase_col(F.col("over_no")))
        .withColumn("bat_legal", F.when(is_legal_for_batter(), 1).otherwise(0))
        .withColumn("bowl_legal", F.when(is_legal_for_bowler(), 1).otherwise(0))
        .withColumn("runs_conceded", runs_conceded_col())
        .withColumn("bowler_wkt", F.when(is_bowler_wicket(), 1).otherwise(0))
        .withColumn("wkt_lost", F.when(F.col("wicket_fell") == True, 1).otherwise(0))
        .withColumn("four", F.when(F.col("runs_batter") == 4, 1).otherwise(0))
        .withColumn("six",  F.when(F.col("runs_batter") == 6, 1).otherwise(0))
    )

    # -------- Batting by team_id --------
    bat = (d2
           .where(F.col("batting_team_id").isNotNull())
           .groupBy("season","batting_team_id","phase")
           .agg(
               F.sum("runs_total").alias("runs_scored"),
               F.sum("bat_legal").alias("balls_faced"),
               F.col("four").alias("fours"),
               F.col("six").alias("sixes"),
               F.col("wkt_lost").alias("wickets_lost")
           )
           .withColumn("overs_faced", F.col("balls_faced") / F.lit(6.0))
           .withColumn("run_rate", safe_div(F.col("runs_scored") * F.lit(6.0), F.col("balls_faced")))
           .withColumnRenamed("batting_team_id","team_id")
        )
    # -------- Bowling by team_id --------

    bowl = (d2
            .where(F.col("bowling_team_id").isNotNull())
            .groupBy("season","bowling_team_id","phase")
            .agg(
                F.sum("runs_conceded").alias("runs_conceded"),
                F.sum("bowl_legal").alias("balls_bowled"),
                F.sum("bowler_wkt").alias("wkts_taken"),
            )
            .withColumn("overs_bowled", F.col("balls_bowled") / F.lit(6.0))
            .withColumn("economy", safe_div(F.col("runs_conceded") * F.lit(6.0), F.col("balls_bowled")))
            .withColumnRenamed("bowling_team_id", "team_id")
    )

    # Combine (full outer to keep phases if one side missing)
    gold = (bat.join(bowl, ["season", "team_id", "phase"], "full")
              .fillna(0, subset=[
                  "runs_scored","balls_faced","fours","sixes","wickets_lost",
                  "runs_conceded","balls_bowled","wkts_taken"
              ])
              .withColumn("run_rate",  F.when(F.col("balls_faced") == 0, None).otherwise(F.col("run_rate")))
              .withColumn("economy",   F.when(F.col("balls_bowled") == 0, None).otherwise(F.col("economy")))
           )
    
    # Add team name (optional)
    try:
        dim_team_path = root / cfg["tables"]["dim_team"]["target"]
        dim_team = spark.read.format(fmt).load(str(dim_team_path)).select("team_id", F.col("name_canonical").alias("team_name"))
        gold = gold.join(dim_team, "team_id", "left")
    except Exception:
        pass

    out = gold.select(
        "season", "team_id", "team_name", "phase",
        "runs_scored","balls_faced","overs_faced","run_rate","fours","sixes","wickets_lost",
        "runs_conceded","balls_bowled","overs_bowled","economy","wkts_taken"
    )

    print("\n=== gold_team_phase_season sample ===")
    out.orderBy("season","team_name","phase").show(30, truncate=False)

    if write_out and not sample_only:
        target = root / cfg["gold"]["tables"]["team_phase_season"]
        (out.repartition(1, "season")
            .write.mode("overwrite")
            .format(cfg["gold"]["format"])
            .save(str(target)))
        print(f"\n✓ Wrote gold_team_phase_season to: {target}")

    spark.stop()

if __name__ == "__main__":
    # preview first; then write full
    main(write_out=False, sample_only=True)
