# etl/silver_transform/enrich_deliveries_teams.py
from pathlib import Path
from pyspark.sql import functions as F, types as T
from etl.common import get_spark, load_yml, project_root
from etl.silver_transform.register_resolver import make_normalizer

def main(write_out: bool = True, sample_only: bool = True):
    root = project_root()
    cfg_silver= load_yml(str(root / "configs" / "silver_config.yml"))
    fmt = cfg_silver["storage"]["format"]

    spark = get_spark("silver-enrich-deliveries-teams")

    # Load dim_team and explode keys for a clean join
    dim_path = root / cfg_silver["tables"]["dim_team"]["target"]
    dim = spark.read.format(fmt).load(str(dim_path))

    # Explode aliases_norm so each normalized key maps to team_id
    keys = (dim
        .select(
            "team_id",
            F.explode(
                F.array_union(F.array(F.col("name_canonical_norm")), F.col("aliases_norm"))
            ).alias("key_norm")
        )
    )

    # Read Silver deliveries
    d_path = root / cfg_silver["tables"]["deliveries"]["target"]
    df = spark.read.format(fmt).load(str(d_path))

    # Normalizer for join keys
    norm_rules = cfg_silver["team_normalization"]
    normalizer = make_normalizer(norm_rules)
    norm_udf = F.udf(lambda s: normalizer(s), T.StringType())

    # Join for batting_team
    d1 = (df
        .withColumn("bat_key", norm_udf(F.col("batting_team")))
        .withColumn("bowl_key", norm_udf(F.col("bowling_team")))
        .join(keys.withColumnRenamed("team_id", "batting_team_id")
                  .withColumnRenamed("key_norm", "bat_key"),
              on="bat_key", how="left")
        .join(keys.withColumnRenamed("team_id", "bowling_team_id")
                  .withColumnRenamed("key_norm", "bowl_key"),
              on="bowl_key", how="left")
    )

    out = d1.drop("bat_key", "bowl_key")

    print("\n=== deliveries with team IDs (preview) ===")
    out.select("match_id","inning_no","over_no","ball_in_over",
               "batting_team","batting_team_id","bowling_team","bowling_team_id").show(20, truncate=False)

    if write_out and not sample_only:
        target = root / cfg_silver["tables"]["deliveries"]["target"]
        parts  = cfg_silver["tables"]["deliveries"]["partition_columns"]
        (out
         .repartition(1, "season")
         .write.mode("overwrite")
         .partitionBy(*parts)
         .format(fmt)
         .save(str(target)))
        print(f"\n✓ Updated Silver deliveries with team IDs at: {target}")

    spark.stop()

if __name__ == "__main__":
    main(write_out=True, sample_only=False)
