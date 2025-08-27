from pathlib import Path
from ruamel.yaml import YAML
from pyspark.sql import SparkSession
import os

def load_yml(path_str:str) -> dict:
    """ Load a yaml file into ptyhon dict"""
    yaml = YAML(typ="safe")
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Yaml not found {path}")
    with path.open("r",encoding="utf-8") as f:
        return yaml.load(f) or {}

def get_spark(app_name: str = "cricket-pipeline") -> SparkSession:

    """
    Create a local SparkSession suitable for development.
    Key ideas:
    - Arrow enabled for fast Pandas conversion (handy in notebooks).
    - shuffle.partitions small for local runs (speed).
    - wholeTextFiles not required; Spark can read multiline JSON.
    """

    root = Path(__file__).resolve().parents[1]
    event_dir = root / "logs" / "spark-events"
    event_dir.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName(app_name) # type: ignore
        .master("local[*]") ## using all the cores
        .config("spark.sql.execution.arrow.pyspark.enabled","true")
        .config("spark.sql.shuffle.partitions","8")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.hadoop.io.file.buffer.size", "65536")
        # Disable Hadoop native libraries to fix Windows compatibility issues
        .config("spark.hadoop.io.native.lib.available", "false")
        # UI port (use 4040 or another single value you prefer)
        .config("spark.ui.port", "4040")
        # Enable event logging so History Server can show finished apps
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", f"file:///{str(event_dir).replace(os.sep, '/')}")
        .getOrCreate()  
    )
    return spark

def project_root() -> Path:
    """Return repo root assuming this file lives under ./etl/."""
    return Path(__file__).resolve().parents[1]