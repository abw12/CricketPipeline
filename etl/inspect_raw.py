from ast import alias, expr
from math import trunc
from common import project_root, load_yml, get_spark
from pathlib import Path
from pyspark.sql import functions as F


def main():
    root = project_root()
    cfg = load_yml(str(root/"configs"/"bronze_config.yml"))
    raw_dir = cfg["source"]["raw_path"] # e.g., data/raw/ipl
    file_glob = cfg["source"].get("file_glob","*.json")
    multi_line = cfg["source"].get("read_options", {}).get("multiline", True)

    # List a handful of files to speed up inspection.
    raw_path = root / raw_dir
    files = sorted(Path(raw_path).glob(file_glob))[:3] # read first 10
    if not files:
        raise FileNotFoundError(f"No JSON files found under {raw_path} with glob {file_glob}")
    
    spark = get_spark("bronze-inspect-raw")

    # Read selected JSONs. Cricsheet files are multiline, so set option accordingly.
    df = (
    spark.read
    .option("multiLine",str(multi_line).lower())
    .json([str(p) for p in files])
    .withColumn("src_file_path", F.input_file_name())
    )

    print("\n=== RAW TOP-LEVEL SCHEMA ===")
    df.printSchema()

    # Peek into 'info' block (match-level metadata)
    info_cols = [
        "info.season",
        "info.officials",
        "info.match_type",
        "info.gender",
        "info.city",
        "info.venue",
        "info.teams",
        "info.dates",
        "info.toss",
        "info.outcome",
        "info.player_of_match",
        "src_file_path",
    ]
    print("\n=== SAMPLE: info columns (first 3 rows) ===")
    df.select(*[F.col(c) for c in info_cols]).show(3, truncate=False)

    # Quick sanity: number of innings and overs in the first file
    # size(innings) tells you how many innings objects exist
    print("\n=== SAMPLE: innings size and first innings' team ===")
    df_innings = df.select(
        F.size("innings").alias("n_innings"),
        F.expr("CASE WHEN size(innings) > 0 THEN innings[0].team ELSE NULL END").alias("inning1_team"),
        "src_file_path"
    )
    df_innings.show(5,truncate=False)
    # Explode first level to see over/delivery nesting (just to visualize structure)
    # NOTE: This is an INSPECTION, not our final flattening logic.

    df_ex = (
        df
        .withColumn("inning_pos", F.sequence(F.lit(0), F.size("innings") -1 ))
        .withColumn("inning_pos", F.explode("inning_pos"))
        .withColumn("inning", F.col("innings")[F.col("inning_pos")])
        .withColumn("team_batting", F.col("inning.team"))
        .withColumn("overs", F.col("inning.overs"))
        .drop("innings")
        .withColumn("overs_pos",F.sequence(F.lit(0),F.size("overs") -1))
        .withColumn("overs_pos",F.explode("overs_pos"))
        .withColumn("over_struct", F.col("overs")[F.col("overs_pos")])
        .withColumn("deliveries", F.col("over_struct.deliveries"))
    )
    print("\n=== INSPECT: first few (inning, over) pairs with team_batting ===")
    df_ex.select("team_batting","overs_pos","src_file_path").show(10,truncate=False)

     # Count total deliveries in the sample files (just to get a feel)
    df_deliveries = df_ex.withColumn("d_pos", F.sequence(F.lit(0),F.size("deliveries")-1)) \
        .withColumn("d_pos",F.explode("d_pos"))
    
    print("\n=== SANITY: approximate deliveries in these sample files ===")
    print(f"Sample files: {len(files)}")
    print(f"Approx deliveries rows (not fully flattened yet): {df_deliveries.count()}")

    # Print the first over’s first two deliveries fields (batter, bowler, non_striker, runs, extras, wickets)
    first_over = (
        df_ex
        .filter(F.col("overs_pos") == 0)  # first over in each match
        .orderBy("src_file_path")        # deterministic pick
        .limit(1)                        # just one row to read clearly
    )

    # Shortcuts to the first two deliveries within the 'deliveries' array
    d1 = F.col("deliveries")[0]  # first ball
    d2 = F.col("deliveries")[1]  # second ball

    # Select nested fields from the first two balls
    out = (
        first_over.select(
            "src_file_path",
            "team_batting",
            "overs_pos",

            # --- Delivery 1 (first ball) ---
            d1.getField("batter").alias("d1_striker"),
            d1.getField("bowler").alias("d1_bowler"),
            d1.getField("non_striker").alias("d1_non_striker"),
            d1.getField("runs").getField("batter").alias("d1_runs_batter"),
            d1.getField("runs").getField("extras").alias("d1_runs_extras"),
            d1.getField("runs").getField("total").alias("d1_runs_total"),
            # extras struct may be partially missing; coalesce to 0 for readability
            F.coalesce(d1.getField("extras").getField("wides"), F.lit(0)).alias("d1_extra_wides"),
            F.coalesce(d1.getField("extras").getField("noballs"), F.lit(0)).alias("d1_extra_noballs"),
            # wickets is an array (could be empty); element_at is NULL‑safe
            F.element_at(d1.getField("wickets"), 1).getField("player_out").alias("d1_player_out"),
            F.element_at(d1.getField("wickets"), 1).getField("kind").alias("d1_wicket_kind"),

            # --- Delivery 2 (second ball) ---
            d2.getField("batter").alias("d2_striker"),
            d2.getField("bowler").alias("d2_bowler"),
            d2.getField("non_striker").alias("d2_non_striker"),
            d2.getField("runs").getField("batter").alias("d2_runs_batter"),
            d2.getField("runs").getField("extras").alias("d2_runs_extras"),
            d2.getField("runs").getField("total").alias("d2_runs_total"),
            F.coalesce(d2.getField("extras").getField("wides"), F.lit(0)).alias("d2_extra_wides"),
            F.coalesce(d2.getField("extras").getField("noballs"), F.lit(0)).alias("d2_extra_noballs"),
            F.element_at(d2.getField("wickets"), 1).getField("player_out").alias("d2_player_out"),
            F.element_at(d2.getField("wickets"), 1).getField("kind").alias("d2_wicket_kind"),
        )
    )

    out.show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()