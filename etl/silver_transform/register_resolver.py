from pathlib import Path
from pydoc import resolve
from typing import Dict, Tuple
import unicodedata, re
from pyspark.sql import functions as F, types as T
from etl.common import get_spark, load_yml, project_root


# ---------- Normalization helpers ----------

def normalize_text(s:str,lower=True,
                   strip=True,collapse_spaces=True,
                   remove_diacritics=True, remove_trailing_commas=True,
                   remove_periods=False) -> str:
    if s is None:
        return None

    out = s
    if remove_diacritics:
        out = unicodedata.normalize("NFKD", out)
        out = "".join(ch for ch in out if not unicodedata.combining(ch))
    if lower:
        out = out.lower()
    if strip:
        out = out.strip()
    if remove_trailing_commas:
        out = re.sub(r",+$", "", out).strip()
    if remove_periods:
        out = out.replace(".", "")
    if collapse_spaces:
        out = " ".join(out.split())
    return out

def make_normalizer(rules: dict):
    def _norm(s:str) -> str:
        return normalize_text(
            s,
            lower=rules.get("lower", True),
            strip=rules.get("strip", True),
            collapse_spaces=rules.get("collapse_spaces", True),
            remove_diacritics=rules.get("remove_diacritics", True),
            remove_trailing_commas=rules.get("remove_trailing_commas", True),
            remove_periods=rules.get("remove_periods", False),
        )
    return _norm

def build_player_maps(players_df,aliases_df, colP, colA, normalizer) -> Dict[str,str]:
    """
    Returns a single dict mapping normalized strings to player identifier.
    Keys inserted:
      - normalized unique_name -> id
      - normalized name        -> id
      - normalized alias       -> id
    """
    mapping: Dict[str, str] = {}

    # players: unique_name + name
    rowsP = players_df.select(colP["id"], colP["unique_name"], colP["name"]).collect()
    for r in rowsP:
        pid = r[colP["id"]]
        if r[colP["unique_name"]]:
            mapping[normalizer(r[colP["unique_name"]])] = pid
        if r[colP["name"]]:
            mapping[normalizer(r[colP["name"]])] = pid

    # aliases: alias
    rowsA = aliases_df.select(colA["id"], colA["alias"]).collect()
    for r in rowsA:
        pid = r[colA["id"]]
        alias = r[colA["alias"]]
        if alias:
            mapping[normalizer(alias)] = pid

    return mapping

def make_udf_player_resolver(b_map, normalizer):
    @F.udf(returnType=T.StringType())
    def resolve(name:str) -> str:
        if name is None:
            return None
        n = normalizer(name)
        return b_map.value.get(n) #b_map is the broadcast dictionary on the executor
    return resolve

def main_preview():
    root = project_root()
    cfg_reg = load_yml(str(root / "configs" / "register_config.yml"))
    cfg_silver = load_yml(str(root / "configs" / "silver_config.yml"))
    cfg_bronze = load_yml(str(root / "configs" / "bronze_config.yml"))
    fmt = cfg_silver["storage"]["format"]

    spark = get_spark("register-preview-v2")

    # load Register CSVs
    ppath = root / cfg_reg["paths"]["players"]
    apath = root / cfg_reg["paths"]["player_aliases"]
    colsP = cfg_reg["columns"]["players"]
    colsA = cfg_reg["columns"]["player_aliases"]

    players =  spark.read.option("header",True).csv(str(ppath))
    aliases = spark.read.option("header",True).csv(str(apath))

    normalizer = make_normalizer(cfg_reg["name_normalization"])

    # Build map: (unique_name/name/alias) -> identifier
    player_map = build_player_maps(players,aliases,colsP,colsA,normalizer)

    # Broadcast + UDF
    b_player_map = spark.sparkContext.broadcast(player_map)
    resolve_player = make_udf_player_resolver(b_player_map, normalizer)

    # Preview: apply on a small Bronze deliveries sample
    bronze_deliveries_path = root / cfg_bronze["tables"]["deliveries"]["target_path"]
    d_bz = spark.read.format(fmt).load(str(bronze_deliveries_path))
    sample = d_bz.select("match_id","season","striker","non_striker","bowler").limit(25)

    preview = (sample
               .withColumn("striker_id", resolve_player(F.col("striker")))
               .withColumn("non_striker_id",resolve_player(F.col("non_striker")))
               .withColumn("bowler_id", resolve_player(F.col("bowler")))
               )
    
    print("\n=== Register mapping preview (first 25) ===")
    preview.show(25, truncate=False)

    # Coverage stats
    cov = (preview
           .select(
               (F.col("striker_id").isNotNull()).cast("int").alias("striker_mapped"),
               (F.col("non_striker_id").isNotNull()).cast("int").alias("non_striker_mapped"),
               (F.col("bowler_id").isNotNull()).cast("int").alias("bowler_mapped")
           )
           .agg(*(F.avg(c).alias(c+"_pct") for c in ["striker_mapped","non_striker_mapped","bowler_mapped"])))
    print("\n=== Coverage ===")
    cov.show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main_preview()