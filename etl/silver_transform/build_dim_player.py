from ast import alias
from pathlib import Path
from pyspark.sql import functions as F, types as T
from etl.common import get_spark, load_yml, project_root
from etl.silver_transform.register_resolver import make_normalizer

def main(write_out:bool=True):
    root = project_root()
    cfg_reg   = load_yml(str(root / "configs" / "register_config.yml"))
    cfg_silver= load_yml(str(root / "configs" / "silver_config.yml"))

    spark = get_spark("silver-build-dim-player")

    ppath = root / cfg_reg["paths"]["players"]
    apath = root / cfg_reg["paths"]["player_aliases"]

    colsP = cfg_reg["columns"]["players"]
    colsA = cfg_reg["columns"]["player_aliases"]

    normalizer = make_normalizer(cfg_reg["name_normalization"])

    players = (spark.read.option("header",True).csv(str(ppath))
               .select(
                    F.col(colsP["id"]).alias("player_id"),
                    F.col(colsP["name"]).alias("name"),
                    F.col(colsP["unique_name"]).alias("unique_name"),
                    F.col(colsP.get("id_cricinfo", None)).alias("id_cricinfo") if colsP.get("id_cricinfo") else F.lit(None).alias("id_cricinfo"),
                    F.col(colsP.get("id_cricbuzz", None)).alias("id_cricbuzz") if colsP.get("id_cricbuzz") else F.lit(None).alias("id_cricbuzz"),
               ))
    aliases = (spark.read.option("header",True).csv(str(apath))
               .select(
                    F.col(colsA["id"]).alias("player_id"),
                    F.col(colsA["alias"]).alias("alias")
               ))
    # Normalize display name and unique_name, and collect aliases
    norm_udf = F.udf(lambda s:normalizer(s),T.StringType())

    players_norm = (players
                    .withColumn("name_norm", norm_udf(F.col("name")))
                    .withColumn("unique_name_norm", norm_udf(F.col("unique_name")))
                )
    aliases_norm = (aliases
        .withColumn("alias_norm", norm_udf(F.col("alias")))
        .groupBy("player_id").agg(F.collect_set("alias_norm").alias("aliases_norm"))
    )

    dim = (players_norm
            .join(aliases_norm,"player_id","left")
            .withColumn("aliases_norm",F.coalesce(F.col("aliases_norm"), F.array().cast("array<string>")))
            # make a unified set of all keys that should map to this player
            .withColumn("all_keys_norm",
                        F.array_union(
                                F.array(F.col("name_norm"),F.col("unique_name_norm")),
                                F.col("aliases_norm")
                            )
                        )
                        .select(
                            "player_id", "name", "unique_name", "id_cricinfo", "id_cricbuzz",
                            "name_norm", "unique_name_norm", "aliases_norm", "all_keys_norm"
                        )
    )

    print("\n=== dim_player sample ===")
    dim.show(10, truncate=False)
    dim.printSchema()

    if write_out:
        target = root / cfg_silver["tables"]["dim_player"]["target"]
        (dim
         .repartition(1)
         .write.mode("overwrite")
         .format(cfg_silver["storage"]["format"])
         .save(str(target)))
        print(f"\n✓ Wrote dim_player to: {target}")

    spark.stop()

if __name__ == "__main__":
    main(write_out=True)

