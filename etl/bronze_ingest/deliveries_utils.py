from tkinter.tix import COLUMN
from typing import Optional
import hashlib
from h11 import Data
from pyspark.sql import functions as F, types as T, Column, DataFrame

def sha1_16(s:str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

@F.udf(returnType=T.StringType())
def udf_sha1_16(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return sha1_16(s)

def compute_match_id_from_columns()-> Column:
    """
    Build a match_id from the columns we selected in C1:
    season|competition|team1|team2|venue|match_start_date  (lowercased, trimmed)
    """

    key = F.concat_ws(
        "|",
        F.lower(F.coalesce(F.col("season").cast("string"),F.lit(""))),
        F.lower(F.coalesce(F.col("team1"), F.lit(""))),
        F.lower(F.coalesce(F.col("team2"), F.lit(""))),
        F.lower(F.coalesce(F.col("venue"), F.lit(""))),
        F.lower(F.coalesce(F.col("match_start_date"), F.lit(""))),
    )
    return udf_sha1_16(key)

def compute_delivery_id_expr()->Column:
    return F.concat_ws(
        "-",
        F.col("match_id"),
        F.format_string("%02d", F.col("inning_no")),
        F.format_string("%02d", F.col("over_no")),
        F.format_string("%02d", F.col("ball_in_over")),
    )

def add_lineage_from_struct(df:DataFrame,struct_col:str) -> DataFrame:
    """
    Add lineage columns using a provided struct column to hash (e.g., 'delivery').
    """
    df1 = df.withColumn("src_file_name", F.element_at(F.split(F.col("src_file_path"), r"[\\/]+"), -1))
    df2 = df1.withColumn("src_ingest_ts", F.current_timestamp())
    # Convert to_json first and then handle null with coalesce to empty json string
    json_blob = F.coalesce(F.to_json(F.col(struct_col)), F.lit("{}"))
    df3 = df2.withColumn("src_record_hash", udf_sha1_16(json_blob))
    return df3