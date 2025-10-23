from pathlib import Path
from pyspark.sql import functions as F, types as T
import hashlib

from etl.common import get_spark, load_yaml, project_root
from etl.silver_transform.register_resolver import make_normalizer  # reuse the same normalizer factory

def sha1_12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def main(write_out: bool = True):
    root = project_root()
    cfg_silver = load_yaml(str(root / "configs" / "silver_config.yml"))
    cfg_bronze = load_yaml(str(root / "configs" / "bronze_config.yml"))
    fmt = cfg_silver["storage"]["format"]

    spark = get_spark("silver-build-dim-team")

    # Read Silver matches & deliveries (or Bronze if you prefer; Silver is cleaner)
    m_path = root / cfg_silver["tables"]["matches"]["target"]
    d_path = root / cfg_silver["tables"]["deliveries"]["target"]
    matches = spark.read.format(fmt).load(str(m_path))
    deliveries = spark.read.format(fmt).load(str(d_path))
    # Pull all team name columns we see in facts
    
    team_cols = [
        F.col("team1").alias("team"),
        F.col("team2").alias("team"),
        F.col("toss_winner").alias("team"),
        F.col("winner").alias("team"),
        F.col("first_batting_team").alias("team"),
        F.col("second_batting_team").alias("team"),
    ]

    m_teams = matches.select(*team_cols)

    d_teams = deliveries.select(
        F.col("batting_team").alias("team"),
        F.col("bowling_team").alias("team"),
    )

    raw_teams = m_teams.unionByName(d_teams).filter(F.col("team").isNotNull()).distinct()

    # normalization + overrides
    norm_rules = cfg_silver["team_normalization"]
    normalizer = make_normalizer(norm_rules)
    norm_udf = F.udf(lambda s: normalizer(s), T.StringType())

    overrides = cfg_silver.get("team_overrides", {}) or {}
    # Normalize override keys & values once (so matching is robust)
    overrides_norm = { normalizer(k) : overrides[k] for k in overrides}
    # Also normalize override values to use as canonical keys when generating IDs
    override_canon_norm = { normalizer(k) : normalizer(v) for k,v in overrides.items()}

    # Apply normalization
    teams_norm = (raw_teams
                  .withColumn("team_norm", norm_udf(F.col("team")))
    )

    # Apply overrides: if team_norm appears as a key, use override's canonical display name, else original
    def choose_canonical(team: str, team_norm: str) -> str:
        # If there is an override for this normed key, return the override display name
        if team_norm in overrides_norm:
            return overrides_norm[team_norm]
        # else use the original string as canonical display name
        return team
    
    choose_canon_udf = F.udf(lambda team, team_norm: choose_canonical(team,team_norm),T.StringType())
    teams_canon = (teams_norm
                   .withColumn("name_canonical", choose_canon_udf(F.col("team"),F.col("team_norm")))
                   .withColumn("name_canonical_norm", norm_udf(F.col("name_canonical")))
                   )
    dim =(teams_canon
          .groupBy("name_canonical","name_canonical_norm")
          .agg(F.collect_set("team_norm").alias("aliases_norm"))
    )
    # Deterministic team_id from canonical normalized name
    mk_id = F.udf(lambda s : "team_" + sha1_12(s), T.StringType())
    dim = dim.withColumn("team_id", mk_id(F.col("name_canonical_norm")))

    # Arrange columns
    dim = dim.select(
        "team_id",
        "name_canonical",
        "name_canonical_norm",
        "aliases_norm"
    )
    print("\n=== dim_team sample ===")
    dim.orderBy("name_canonical").show(20, truncate=False)
    dim.printSchema()

    if write_out:
        target = root / cfg_silver["tables"]["dim_team"]["target"]
        (dim
         .repartition(1)
         .write.mode("overwrite")
         .format(cfg_silver["storage"]["format"])
         .save(str(target)))
        print(f"\n✓ Wrote dim_team to: {target}")

    spark.stop()

if __name__ == "__main__":
    main(write_out=True)