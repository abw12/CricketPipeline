from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

from etl.common import PipelineContext, require_spark, show_preview, write_table
from etl.silver_transform.register_resolver import make_normalizer


def build_dim_player(players: DataFrame, aliases: DataFrame, cfg_register: dict) -> DataFrame:
    cols_p = cfg_register["columns"]["players"]
    cols_a = cfg_register["columns"]["player_aliases"]
    normalizer = make_normalizer(cfg_register["name_normalization"])
    norm_udf = F.udf(lambda s: normalizer(s), T.StringType())

    selected_players = players.select(
        F.col(cols_p["id"]).alias("player_id"),
        F.col(cols_p["name"]).alias("name"),
        F.col(cols_p["unique_name"]).alias("unique_name"),
        F.col(cols_p["id_cricinfo"]).alias("id_cricinfo")
        if cols_p.get("id_cricinfo")
        else F.lit(None).alias("id_cricinfo"),
        F.col(cols_p["id_cricbuzz"]).alias("id_cricbuzz")
        if cols_p.get("id_cricbuzz")
        else F.lit(None).alias("id_cricbuzz"),
    )
    selected_aliases = aliases.select(
        F.col(cols_a["id"]).alias("player_id"),
        F.col(cols_a["alias"]).alias("alias"),
    )

    players_norm = (
        selected_players.withColumn("name_norm", norm_udf(F.col("name")))
        .withColumn("unique_name_norm", norm_udf(F.col("unique_name")))
    )
    aliases_norm = (
        selected_aliases.withColumn("alias_norm", norm_udf(F.col("alias")))
        .groupBy("player_id")
        .agg(F.collect_set("alias_norm").alias("aliases_norm"))
    )

    empty_array = F.array().cast(T.ArrayType(T.StringType()))
    return (
        players_norm.join(aliases_norm, "player_id", "left")
        .withColumn("aliases_norm", F.coalesce(F.col("aliases_norm"), empty_array))
        .withColumn(
            "all_keys_norm",
            F.array_union(F.array(F.col("name_norm"), F.col("unique_name_norm")), F.col("aliases_norm")),
        )
        .select(
            "player_id",
            "name",
            "unique_name",
            "id_cricinfo",
            "id_cricbuzz",
            "name_norm",
            "unique_name_norm",
            "aliases_norm",
            "all_keys_norm",
        )
    )


def run(
    context: Optional[PipelineContext] = None,
    spark: Optional[SparkSession] = None,
    sample_only: bool = False,
    write_out: bool = True,
    preview: bool = False,
) -> None:
    context = context or PipelineContext.load()
    spark, should_stop = require_spark("silver-build-dim-player", spark)
    try:
        players = spark.read.option("header", True).csv(str(context.resolve(context.register["paths"]["players"])))
        aliases = spark.read.option("header", True).csv(
            str(context.resolve(context.register["paths"]["player_aliases"]))
        )
        dim = build_dim_player(players, aliases, context.register)
        if sample_only:
            dim = dim.limit(100)
        if preview:
            show_preview(dim, "dim_player", 10)
        if write_out:
            target = write_table(dim, context.silver_table_path("dim_player"), context.silver_format)
            print(f"Wrote dim_player to: {target}")
    finally:
        if should_stop:
            spark.stop()


def main(write_out: bool = True) -> None:
    run(write_out=write_out, preview=True)


if __name__ == "__main__":
    main(write_out=True)


