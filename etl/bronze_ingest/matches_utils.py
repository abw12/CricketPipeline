from pathlib import Path
import hashlib
import sys
import os

# Add parent directory to Python path to allow module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional
from pyspark.sql import functions as F, types as T, DataFrame, Column

# --------- Deterministic ID ----------

def sha1_16(s : str) -> str:
    """Return first 16 hex chars of a SHA1 for stable, short IDs."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

@F.udf(returnType=T.StringType())
def udf_sha1_16(s:Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return sha1_16(s)

def compute_match_id_expr() -> Column:
    """
    Build a Spark Column expression that produces a stable match_id.
    Priority (documented in configs/bronze_config.yml):
    - If we can form a good string from season + teams + venue + date, hash it.
    - You can extend this later if you discover Cricsheet has an explicit id for some formats.
    """
    # Lower/trim to avoid case/space drift; join with pipes to get a deterministic key
    key = F.concat_ws(
        "|",
        F.lower(F.coalesce(F.col("season").cast("string"),F.lit(""))),
        F.lower(F.coalesce(F.col("team1"),F.lit(""))),
        F.lower(F.coalesce(F.col("team2"),F.lit(""))),
        F.lower(F.coalesce(F.col("venue"),F.lit(""))),
        F.lower(F.coalesce(F.col("match_start_date"),F.lit(""))),
    )
    return udf_sha1_16(key) 

# ------- Lineage columns -----

def add_lineage(df: DataFrame, src_path_col: str="src_file_path") -> DataFrame:
    
    """
    Add lineage columns:
      - src_file_name: last path segment of src_file_path
      - src_ingest_ts: current_timestamp
      - src_record_hash: stable hash of the selected record (here we hash the 'info' struct)
    """
    # src_file_name from src_file_path
    df1 = df.withColumn("src_file_name", F.element_at(F.split(F.col(src_path_col), r"[\\/]+"),-1))
    df2 = df1.withColumn("src_ingest_ts",F.current_timestamp())
    # Hash: use a JSON string of info.* to form a deterministic content hash
    # (Null-safe: coalesce to empty JSON when missing)
    json_blob = F.to_json(F.coalesce(F.col("info_struct"),F.lit(F.create_map())))
    df3 = df2.withColumn("src_record_hash",udf_sha1_16(json_blob))
    return df3

def ensure_dir(path_str: str) -> None:
    """Create parent directories if they don't exist (for local runs)."""
    
    p = Path(path_str)
    p.mkdir(parents=True, exist_ok=True)


