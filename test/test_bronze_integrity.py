from pathlib import Path
from pyspark.sql import functions as F
import sys
import os

current_dir = os.path.abspath('')
# Add parent directory to Python path to allow module imports
sys.path.append(os.path.abspath(os.path.join(current_dir,'..','..')))

from etl.common import load_yml, project_root

def test_bronze_integrity(spark):
    root = project_root()
    cfg = load_yml(str(root / "configs" / "bronze_config.yml"))
    fmt = cfg["storage"]["format"]

    d_path = root / cfg["tables"]["deliveries"]["target_path"]
    m_path = root / cfg["tables"]["matches"]["target_path"]

    deliveries = spark.read.format(fmt).load(str(d_path))
    matches    = spark.read.format(fmt).load(str(m_path))

    # 1) deliveries per match band
    dpm = deliveries.groupBy("match_id").count().withColumnRenamed("count","c")
    assert dpm.filter((F.col("c") < 150) | (F.col("c") > 400)).count() == 0

    # 2) runs_total equals sum
    assert deliveries.filter(
        F.coalesce(F.col("runs_total"), F.lit(0)) !=
        (F.coalesce(F.col("runs_batter"), F.lit(0)) + F.coalesce(F.col("runs_extras"), F.lit(0)))
    ).count() == 0

    # 3) bowling team sanity
    assert deliveries.filter(
        (F.col("bowling_team") == F.col("batting_team")) |
        (~((F.col("bowling_team") == F.col("team1")) | (F.col("bowling_team") == F.col("team2"))))
    ).count() == 0