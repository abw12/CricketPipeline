# etl/silver_transform/enrich_matches_teams.py
from pathlib import Path
from pyspark.sql import functions as F, types as T
from etl.common import get_spark, load_yml, project_root
from etl.silver_transform.register_resolver import make_normalizer

def main(write_out: bool = True, sample_only: bool = True):
    root = project_root()
    cfg_silver= load_yml(str(root / "configs" / "silver_config.yml"))
    fmt = cfg_silver["storage"]["format"]

    spark = get_spark("silver-enrich-matches-teams")

    dim_path = root / cfg_silver["tables"]["dim_team"]["target"]
    keys =(spark.read.format(fmt).load(str(dim_path))
           .select(
               "team_id",
               F.explode(
                   F.array_union(F.array(F.col("name_canonical_norm")),F.col("aliases_norm"))
               ).alias("key_norm")
           )
        )
    
    m_path = root / cfg_silver["tables"]["matches"]["target"]
    m = spark.read.format(fmt).load(str(m_path))

    norm_rules = cfg_silver["team_normalization"]
    normalizer = make_normalizer(norm_rules)
    norm_udf = F.udf(lambda s: normalizer(s), T.StringType())

    # Prepare keys for all team-like columns
    m2 = (m
      .withColumn("team1_key", norm_udf(F.col("team1")))
      .withColumn("team2_key", norm_udf(F.col("team2")))
      .withColumn("toss_winner_key", norm_udf(F.col("toss_winner")))
      .withColumn("winner_key", norm_udf(F.col("winner")))
      .withColumn("first_key", norm_udf(F.col("first_batting_team")))
      .withColumn("second_key", norm_udf(F.col("second_batting_team")))
    )

        # Join repeatedly (left) to attach IDs
    joined = (m2
      .join(keys.withColumnRenamed("team_id","team1_id").withColumnRenamed("key_norm","team1_key"), "team1_key", "left")
      .join(keys.withColumnRenamed("team_id","team2_id").withColumnRenamed("key_norm","team2_key"), "team2_key", "left")
      .join(keys.withColumnRenamed("team_id","toss_winner_id").withColumnRenamed("key_norm","toss_winner_key"), "toss_winner_key", "left")
      .join(keys.withColumnRenamed("team_id","winner_id").withColumnRenamed("key_norm","winner_key"), "winner_key", "left")
      .join(keys.withColumnRenamed("team_id","first_batting_team_id").withColumnRenamed("key_norm","first_key"), "first_key", "left")
      .join(keys.withColumnRenamed("team_id","second_batting_team_id").withColumnRenamed("key_norm","second_key"), "second_key", "left")
    )

    out = (joined
      .drop("team1_key","team2_key","toss_winner_key","winner_key","first_key","second_key"))
    
    print("\n=== matches with team IDs (preview) ===")
    out.select("match_id","season","team1","team1_id","team2","team2_id",
               "toss_winner","toss_winner_id","winner","winner_id",
               "first_batting_team","first_batting_team_id",
               "second_batting_team","second_batting_team_id") \
       .orderBy("season").show(20, truncate=False)

    if write_out and not sample_only:
        target = root / cfg_silver["tables"]["matches"]["target"]
        parts  = cfg_silver["tables"]["matches"]["partition_columns"]
        (out
         .repartition(1, "season")
         .write.mode("overwrite")
         .partitionBy(*parts)
         .format(fmt)
         .save(str(target)))
        print(f"\n✓ Updated Silver matches with team IDs at: {target}")

    spark.stop()

if __name__ == "__main__":
    main(write_out=True, sample_only=False)
