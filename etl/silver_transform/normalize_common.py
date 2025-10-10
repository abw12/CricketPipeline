from typing import Dict
from pyspark.sql import functions as F, types as T, Column

def normalize_enum(col:Column, mapping: Dict[str,str]) -> Column:
    """
    Lower, trim, and map values using a dict; unknowns become NULL.
    """

    base = F.lower(F.trim(col))
    # Build a case expression from mapping
    expr = None
    for k,v in mapping.items():
        cond = (base == F.lit(k))
        expr = F.when(cond,F.lit(v)) if expr is None else expr.when(cond, F.lit(v))
    return expr.otherwise(F.lit(None)) if expr is not None else base

def parse_date_multi(col: Column, patterns) -> Column:
    """
    Try multiple date patterns; return first non-null.
    """
    out = None
    for p in patterns:
        candidate = F.to_date(col, p)
        out = candidate if out is None else F.coalesce(out, candidate)
    return out #type:ignore

def to_bool(col:Column) -> Column:
    return F.when(col.isNull(),F.lit(None)).otherwise(col.cast(T.BooleanType()))