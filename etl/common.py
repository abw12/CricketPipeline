from pathlib import Path
from ruamel.yaml import YAML
from pyspark.sql import SparkSession

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
    spark = (
        SparkSession.builder
        .appName(app_name) # type: ignore
        .master("local[*]") ## using all the cores
        .config("spark.sql.execution.arrow.pyspark.enabled","true")
        .config("spark.sql.shuffle.partitions","8")
        .getOrCreate()  
    )
    return spark

def project_root() -> Path:
    """Return repo root assuming this file lives under ./etl/."""
    return Path(__file__).resolve().parents[1]