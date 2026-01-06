# etl/gold/build_match_summary.py
from ast import alias
from pathlib import Path
from matplotlib.pyplot import spring
from pyspark.sql import functions as F

from etl.common import get_spark, load_yml, project_root
from etl.gold.metrics_common import is_legal_for_bowler

def main(write_out: bool = True,sample_only: bool = True):
    root = project_root()
    cfg = load_yml(str(root / "configs" / "silver_config.yml"))
    fmt = cfg["storage"]["format"]

    spark = get_spark("gold-match-summary")

    m_path = cfg["tables"]["matches"]["target"]
    d_path = cfg["tables"]["deliveries"]["target"]
    m = spark.read.format(fmt).load(str(m_path))
    d = spark.read.format(fmt).load(str(d_path))

    if sample_only:
        some = [r["match_id"] for r in m.select("match_id").limit(50).collect()]
        m = m.where(F.col("match_id").isin(some))
        d = d.where(F.col("match_id").isin(some))
    
    # inning-level totals from deliveries (facts)
    inn = (d
        .groupBy("match_id", "inning_no")
        .agg(
            F.sum("runs_total").alias("runs"),
            F.sum(F.col("wicket_fell").cast("int")).alias("wkts"),
            F.sum(F.when(is_legal_for_bowler(), 1).otherwise(0)).alias("balls_legal"),
            F.first("batting_team").alias("batting_team"),
            F.first("batting_team_id").alias("batting_team_id"),
        )
    )

    # Pivot innings to columns
    inn_p = (inn
        .groupBy("match_id")
        .pivot("inning_no", [1,2])
        .agg(
            F.first("runs").alias("runs"),
            F.first("wkts").alias("wkts"),
            F.first("balls_legal").alias("balls"),
            F.first("batting_team").alias("bat_team"),
            F.first("batting_team_id").alias("bat_team_id"),
        )
    )

    # Spark names pivot columns like "1_runs", "2_wkts", etc.
    # Build match summary
    out = (m
        .select(
            "match_id","season",
            "team1","team1_id","team2","team2_id",
            "winner","winner_id","result_std",
            "match_start_dt","venue","city",
            "first_batting_team","first_batting_team_id",
            "second_batting_team","second_batting_team_id",
        )
        .join(inn_p, "match_id", "left")
        .withColumn("inning1_runs", F.col("1_runs"))
        .withColumn("inning1_wkts", F.col("1_wkts"))
        .withColumn("inning1_balls", F.col("1_balls"))
        .withColumn("inning2_runs", F.col("2_runs"))
        .withColumn("inning2_wkts", F.col("2_wkts"))
        .withColumn("inning2_balls", F.col("2_balls"))
        .withColumn("chase_success",
            F.when(
                (F.col("winner_id").isNotNull()) &
                (F.col("winner_id") == F.col("second_batting_team_id")),
                F.lit(True)
            ).otherwise(F.lit(False))
        )
        .drop("1_runs","1_wkts","1_balls","2_runs","2_wkts","2_balls",
              "1_bat_team","1_bat_team_id","2_bat_team","2_bat_team_id")
    )

    print("\n=== gold_match_summary sample ===")
    out.orderBy("season","match_start_dt").show(20, truncate=False)

    if write_out and not sample_only:
        target = root / cfg["gold"]["tables"]["match_summary"]
        (out.repartition(1, "season")
            .write.mode("overwrite")
            .format(cfg["gold"]["format"])
            .save(str(target)))
        print(f"\n✓ Wrote gold_match_summary to: {target}")
    spark.stop()

if __name__ == "__main__":
    main(write_out=True, sample_only=False)
