from __future__ import annotations

from pyspark.sql import Column, DataFrame, functions as F


def compute_match_id_expr() -> Column:
    key = F.concat_ws(
        "|",
        F.lower(F.coalesce(F.col("season").cast("string"), F.lit(""))),
        F.lower(F.coalesce(F.col("team1"), F.lit(""))),
        F.lower(F.coalesce(F.col("team2"), F.lit(""))),
        F.lower(F.coalesce(F.col("venue"), F.lit(""))),
        F.lower(F.coalesce(F.col("match_start_date"), F.lit(""))),
    )
    return F.substring(F.sha1(key), 1, 16)


def add_lineage(df: DataFrame, src_path_col: str = "src_file_path") -> DataFrame:
    json_blob = F.coalesce(F.to_json(F.col("info_struct")), F.lit("{}"))
    return (
        df.withColumn("src_file_name", F.element_at(F.split(F.col(src_path_col), r"[\\/]+"), -1))
        .withColumn("src_ingest_ts", F.current_timestamp())
        .withColumn("src_record_hash", F.substring(F.sha1(json_blob), 1, 16))
    )

