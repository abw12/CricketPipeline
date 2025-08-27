from pathlib import Path
from pyspark.sql import functions as F, DataFrame
import sys
import os

# Add parent directory to Python path to allow module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from etl.common import get_spark, load_yml, project_root
from etl.bronze_ingest.deliveries_utils import (
    compute_match_id_from_columns,
    compute_delivery_id_expr,
    add_lineage_from_struct,
)

def write_deliveries(df:DataFrame, cfg, root_path):
    target_path= cfg["tables"]["deliveries"]["target_path"] # e.g., data/processed/bronze/deliveries
    parts= cfg["tables"]["deliveries"]["partition_columns"]
    mode = "overwrite"

    full_path = root_path / target_path

    (
        df
        .repartition(1,"season")  # small local runs; tune later
        .write
        .mode(mode)
        .partitionBy(*parts)
        .format(cfg["storage"]["format"])
        .save(str(full_path))
    )

    return full_path

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
    # out = (
    #     df_ex
    #     .select(
    #         "src_file_path", "season", "venue", "match_start_date",
    #         "team1", "team2", "batting_team", "inning_no", "over_no", "ball_in_over",
    #         F.col("delivery.batter").alias("batter"),
    #         F.col("delivery.non_striker").alias("non_striker"),
    #         F.col("delivery.bowler").alias("bowler"),
    #         F.col("delivery.runs.total").alias("runs_total")
    #     )
    #     .orderBy("src_file_path", "inning_no", "over_no", "ball_in_over")
    # )

    # print("\n=== SAMPLE flattened deliveries (first 20 rows) ===")
    # out.show(20, truncate=False)

    # --- Compute match_id the same way as matches ---
    df_w_match = df_ex.withColumn("match_id",compute_match_id_from_columns())

     # --- Derive bowling_team as the "other" team from team1/team2 ---
    df_w_match = df_w_match.withColumn(
        "bowling_team",
        F.when(F.col("batting_team") == F.col("team1"),F.col("team2")).otherwise(F.col("team1"))
    )

    d = F.col("delivery") # shorthand

    # Safe coalesces for extras (null -> 0)
    def nz(c): return F.coalesce(c.cast("int"),F.lit(0))
    
    # Wicket: pick first wicket if present (NULL-safe)
    w1 = F.element_at(d.getField("wickets"),1)

    shaped = (
        df_w_match
        .select(
            #IDs & partition
            "match_id",
            F.col("season").cast("int").alias("season"),

            # Inning/over/ball
            "inning_no", "over_no", "ball_in_over",

            #Team/Players
            "batting_team","bowling_team",
            d.getField("batter").alias("striker"),
            d.getField("non_striker").alias("non_striker"),
            d.getField("bowler").alias("bowler"),

            # Runs (totals & breakdown)
            d.getField("runs").getField("batter").cast("int").alias("runs_batter"),
            d.getField("runs").getField("extras").cast("int").alias("runs_extras"),
            d.getField("runs").getField("total").cast("int").alias("runs_total"),

            nz(d.getField("extras").getField("byes")).alias("extra_byes"),
            nz(d.getField("extras").getField("legbyes")).alias("extra_legbyes"),
            nz(d.getField("extras").getField("wides")).alias("extra_wides"),
            nz(d.getField("extras").getField("noballs")).alias("extra_noballs"),

            #Wickets
            F.when(w1.isNotNull(),F.lit(True)).otherwise(F.lit(False)).alias("wicket_fell"),
            w1.getField("player_out").alias("wicket_player_out"),
            w1.getField("kind").alias("wicket_kind"),
            w1.getField("fielders").alias("wicket_fielders"),

            #lineage seed
            "src_file_path",
            d.alias("delivery_struct_for_hash") # will hash this for src_record_hash
        )
        # Compute delivery_id
        .withColumn("delivery_id",compute_delivery_id_expr())   
    )
    
    # Add lineage columns; then drop the temp struct
    shaped = (
        add_lineage_from_struct(shaped, "delivery_struct_for_hash")
        .drop("delivery_struct_for_hash")
    )

    print("\n=== SAMPLE shaped deliveries (first 20 rows) ===")
    shaped.orderBy("match_id", "inning_no", "over_no", "ball_in_over").show(20, truncate=False)

    print("\n=== deliveries schema (shaped) ===")
    shaped.printSchema()

    # Write full dataset when ready
    if not sample_only:
        out_dir = write_deliveries(shaped, cfg, root)
        print(f"\n✓ Wrote bronze.deliveries to: {out_dir}")

        # Read-back sanity
        df_back = (
            spark.read
            .format(cfg["storage"]["format"])
            .load(str(out_dir))
        )
        print("\n=== read-back schema ===")
        df_back.printSchema()
        print(f"rows: {df_back.count()}")


    # Keep the Spark UI alive for inspection
    # try:
    #     print("\nSpark UI is available at:", spark.sparkContext.uiWebUrl)
    #     print("Keeping SparkContext alive for 5 minutes to allow UI inspection...")
    #     import time
    #     time.sleep(300)  # 5 minutes
    # except KeyboardInterrupt:
    #     print("Interrupted by user, stopping SparkContext...")
    # finally:
    #     spark.stop()
    #     print("SparkContext stopped.")
    spark.stop()

if __name__ == "__main__":
    main(sample_only=False)