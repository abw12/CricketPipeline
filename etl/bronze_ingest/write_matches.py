from pathlib import Path
import sys
import os
import time

# Add parent directory to Python path to allow module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from h11 import Data
from pyspark.sql import functions as F, DataFrame
from etl.common import get_spark, load_yml, project_root
from etl.bronze_ingest.matches_utils import compute_match_id_expr, add_lineage

def select_match_fields(df_raw:DataFrame) -> DataFrame: 
    """
    Flatten the top-level 'info' struct into our matches columns (Bronze schema).
    Keep strings for dates in Bronze; we'll normalize later in Silver.
    """
    info = F.col("info")

    start_date = F.element_at(info.getField("dates"),1).cast("string")
    end_date   = F.element_at(info.getField("dates"), 2).cast("string")

    teams_arr =info.getField("teams") # array of two names 
    team1 = F.element_at(teams_arr,1)
    team2 = F.element_at(teams_arr,2)

    toss = info.getField("toss")
    outcome = info.getField("outcome")

    # Officials can be at info.officials or info.umpires depending on season format
    # We'll coalesce into one array column for Bronze

    officials_arr = info.getField("officials")
    

    df = (
        df_raw
        .select(
            info.alias("info_struct"),
            F.col("src_file_path"),

            # Core match attributes
            info.getField("season").alias("season"),
            info.getField("match_type").alias("match_type"),
            info.getField("gender").alias("gender"),
            info.getField("venue").alias("venue"),
            info.getField("city").alias("city"),

            start_date.alias("match_start_date"),
            end_date.alias("match_end_date"),

            team1.alias("team1"),
            team2.alias("team2"),

            toss.getField("winner").alias("toss_winner"),
            toss.getField("decision").alias("toss_decision"),

            # Outcome normalization: outcome could be {winner, by: {runs|wickets}} or 'result': 'tie/no result'
            outcome.getField("winner").alias("winner"
                                             ),
            # Compose a simple 'result' string in Bronze (silver can restructure)
            F.when(outcome.getField("by").getField("runs").isNotNull(),F.lit("runs"))
            .when(outcome.getField("by").getField("wickets").isNotNull(),F.lit("wickets"))
            .otherwise(F.lit(None)).alias("result"),

            F.coalesce(
                outcome.getField("by").getField("runs"),
                outcome.getField("by").getField("wickets")
            ).cast("int").alias("result_margin"),

            F.element_at(info.getField("player_of_match"),1).alias("player_of_match"),
            officials_arr.alias("officials"),
        )
        # Deterministic ID
        .withColumn("match_id", compute_match_id_expr())
    )

    # Enforce Bronze column order roughly matching bronze_schemas.yml
    ordered = df.select(
        "match_id",
        "season", "match_type", "gender", "venue", "city",
        "match_start_date", "match_end_date",
        "team1", "team2",
        "toss_winner", "toss_decision", "winner", "result", "result_margin", "player_of_match",
        "officials",
        "info_struct", "src_file_path",  # keep info_struct temporarily for hashing in lineage
    )
    return ordered

def write_matches(df_matches:DataFrame, cfg, root_path):
    target_path = cfg["tables"]["matches"]["target_path"]  # e.g., data/processed/bronze/matches
    full_path = root_path / target_path
    
    # Ensure output directory exists
    from etl.bronze_ingest.matches_utils import ensure_dir
    ensure_dir(str(full_path))

    # partition by season as per config
    parts  = cfg["tables"]["matches"]["partition_columns"]
    mode = "overwrite" # for first run; later you can switch to 'append' with dynamic overwrites

    (
        df_matches
        .repartition(1,"season") # small local runs: 1 file per season partition (tweak later)
        .write
        .mode(mode)
        .partitionBy(*parts)
        .format(cfg["storage"]["format"])
        .save(str(full_path))
    )
    return full_path

def main(sample_only:bool = True):
    root = project_root()
    cfg = load_yml(str(root / "configs" / "bronze_config.yml"))

    raw_dir = root / cfg["source"]["raw_path"]
    file_glob = cfg["source"].get("file_glob","*.json")
    multi_line = cfg["source"].get("read_options",{}).get("multiline",True)

    spark = get_spark("bronze-write-matches")

    files = sorted(Path(raw_dir).glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No JSON files under {raw_dir} with glob {file_glob}")

    # For first run, optionally read a small subset (faster feedback)
    paths = [str(p) for p in (files[:50] if sample_only else files)]

    df_raw = (
        spark.read
        .option("multiLine", str(multi_line).lower())
        .json(paths)
        .withColumn("src_file_path",F.input_file_name())
    )

    df_matches = select_match_fields(df_raw)
    df_matches = add_lineage(df_matches).drop("info_struct") # lineage hash already computed from it

    print("\n=== SAMPLE matches rows (first 5) ===")
    df_matches.show(5, truncate=False)

    print("\n=== matches schema ===")
    df_matches.printSchema()

    # After preview, write full dataset:
    if not sample_only:
        out_dir = write_matches(df_matches, cfg, root)
        print(f"\n✓ Wrote bronze.matches to: {out_dir}")

        # Quick read-back validation (schema + simple count)
        df_back = (
            spark.read
            .format(cfg["storage"]["format"])
            .load(str(out_dir))
        )
        print("\n=== read-back schema ===")
        df_back.printSchema()
        print(f"rows: {df_back.count()}")


    # try:
    #     time.sleep(600)  # 600 seconds = 10 minutes
    # except KeyboardInterrupt:
    #     print("Interrupted by user")
    # finally:
    spark.stop()

if __name__ == "__main__":
    main(sample_only=False)