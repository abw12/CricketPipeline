from pyspark.sql import functions as F, Column

def phase_col(over_no_col:Column):
    """
    T20 phases:
      Powerplay: overs 1 to 6
      Middle:    overs 7 to 15
      Death:     overs 16 to 20
    """
    return (
        F.when(over_no_col.between(1,6),F.lit("PP"))
        .when(over_no_col.between(7,15),F.lit("MID"))
        .otherwise(F.lit("DEATH"))
    )
