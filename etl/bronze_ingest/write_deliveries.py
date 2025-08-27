from pathlib import Path
from pyspark.sql import functions as F
import sys
import os

# Add parent directory to Python path to allow module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from etl.common import get_spark, load_yml, project_root

def main(sample_only:bool=True):
    root = project_root()
    cfg = load_yml(str(root / "configs" / "bronze_config.yml"))

    raw_dir = root / cfg["source"]["raw_path"]
    file_glob = cfg["source"].get("file_glob","*.json")
    multi_line = cfg["source"].get("read_options",{}).get("multiline",True)

    spark = get_spark("bronze-write-deliveries")

    files = sorted(Path(raw_dir).glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No JSON files under {raw_dir} with glob {file_glob}")
    
    paths = [str(p) for p in (files[:50] if sample_only else files)]

    df_raw=(
        spark.read
        .option("multiline",str(multi_line).lower())
        .json(paths)
        .withColumn("src_file_path",F.input_file_name())
    )

     # --- Pull small match-level bits we’ll reuse later ---
    info=F.col("info")
    teams=info.getField("teams")
    team1=F.element_at(teams,1).alias("team1")
    team2=F.element_at(teams,2).alias("team2")
    season = info.getField("season").cast("int").alias("season")
    venue = info.getField("venue").alias("venue")
    start_date = F.element_at(info.getField("dates"),1).cast("string").alias("match_start_date")
    # --- Explode innings → overs → deliveries ---
    # posexplode gives us both the array index (position) and the element (value)
    df_ex = (
        df_raw
        .select(
            "src_file_path",
            info.alias("info_struct"),
            season, venue, start_date,
            team1, team2,
            F.posexplode_outer("innings").alias("inning_pos", "inning")
        )
        .withColumn("inning_no", (F.col("inning_pos") + F.lit(1)).cast("int"))
        .withColumn("batting_team", F.col("inning.team"))
        .withColumn("overs", F.col("inning.overs"))
        .drop("inning")
        # Using .select("*", ...) to keep all existing columns while adding the exploded columns
        .select(
            "*", 
            F.posexplode_outer("overs").alias("over_pos", "over_struct")
        )
        .drop("overs")
        # Some seasons include 'over' inside the struct; fall back to position+1
        .withColumn("over_no",
            F.coalesce(F.col("over_struct.over").cast("int"), (F.col("over_pos") + F.lit(1)).cast("int"))
        )
        .select(
            "*",
            F.posexplode_outer(F.col("over_struct.deliveries")).alias("ball_pos", "delivery")
        )
        .drop("over_struct")
        .withColumn("ball_in_over", (F.col("ball_pos") + F.lit(1)).cast("int"))
        .drop("ball_pos")
    )
    # --- Preview a few deliveries with minimal fields ---
    out = (
        df_ex
        .select(
            "src_file_path", "season", "venue", "match_start_date",
            "team1", "team2", "batting_team", "inning_no", "over_no", "ball_in_over",
            F.col("delivery.batter").alias("batter"),
            F.col("delivery.non_striker").alias("non_striker"),
            F.col("delivery.bowler").alias("bowler"),
            F.col("delivery.runs.total").alias("runs_total")
        )
        .orderBy("src_file_path", "inning_no", "over_no", "ball_in_over")
    )

    print("\n=== SAMPLE flattened deliveries (first 20 rows) ===")
    out.show(20, truncate=False)

    # Keep the Spark UI alive for inspection
    try:
        print("\nSpark UI is available at:", spark.sparkContext.uiWebUrl)
        print("Keeping SparkContext alive for 5 minutes to allow UI inspection...")
        import time
        time.sleep(300)  # 5 minutes
    except KeyboardInterrupt:
        print("Interrupted by user, stopping SparkContext...")
    finally:
        spark.stop()
        print("SparkContext stopped.")

if __name__ == "__main__":
    main(sample_only=True)