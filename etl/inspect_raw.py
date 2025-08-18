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
    files = sorted(Path(raw_path).glob(file_glob))[:10] # read first 10
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

    spark.stop()

if __name__ == "__main__":
    main()