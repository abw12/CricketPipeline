from pyspark.sql import functions as F

from etl.common import load_yml, project_root

def test_bronze_integrity(spark):
    root = project_root()
    cfg = load_yml(str(root / "configs" / "bronze_config.yml"))
    fmt = cfg["storage"]["format"]

    d_path = root / cfg["tables"]["deliveries"]["target_path"]
    m_path = root / cfg["tables"]["matches"]["target_path"]

    deliveries = spark.read.format(fmt).load(str(d_path))
    matches    = spark.read.format(fmt).load(str(m_path))

    # 1) required IDs are present and unique where expected
    assert matches.filter(F.col("match_id").isNull()).count() == 0
    assert deliveries.filter(F.col("match_id").isNull() | F.col("delivery_id").isNull()).count() == 0
    assert deliveries.groupBy("delivery_id").count().filter(F.col("count") > 1).count() == 0

    # 2) every delivery belongs to a known match
    assert deliveries.select("match_id").distinct().join(
        matches.select("match_id").distinct(), "match_id", "left_anti"
    ).count() == 0

    # 3) runs_total equals sum
    assert deliveries.filter(
        F.coalesce(F.col("runs_total"), F.lit(0)) !=
        (F.coalesce(F.col("runs_batter"), F.lit(0)) + F.coalesce(F.col("runs_extras"), F.lit(0)))
    ).count() == 0

    # 4) bowling team sanity against match teams
    match_teams = matches.select("match_id", "team1", "team2")
    deliveries_with_teams = deliveries.join(match_teams, "match_id", "left")
    assert deliveries_with_teams.filter(
        (F.col("bowling_team") == F.col("batting_team"))
        | (~((F.col("bowling_team") == F.col("team1")) | (F.col("bowling_team") == F.col("team2"))))
    ).count() == 0
