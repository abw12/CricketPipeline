from pyspark.sql import functions as F, types as T

# -----------------------------
# Rules (documented choices)
# -----------------------------
# Batting balls faced:
#   - Exclude wides (do NOT count towards balls faced)
#   - Include no-balls (common stats convention; wides are the only exclusion)
def is_legal_for_batter():
    return (F.col("extra_wides") == 0)

# Bowling balls:
#   - Legal balls = exclude wides and no-balls (do not advance the over)
def is_legal_for_bowler():
    return ((F.col("extra_wides") == 0) & (F.col("extra_noballs") == 0))

# Bowling runs conceded:
#   - Do NOT count byes/leg-byes against the bowler
def runs_conceded_col():
    return (F.col("runs_total") - F.col("extra_byes") - F.col("extra_legbyes")).cast("int")

# Wickets credited to bowler (exclude run out, retired, etc.)
BOWLER_WICKET_KINDS = {
    "bowled", "caught", "lbw", "stumped", "hit wicket", "caught and bowled", "hit the ball twice"  # conservative set
}
def is_bowler_wicket():
    return (
        (F.col("wicket_fell") == True) &
        F.lower(F.col("wicket_kind")).isin([k.lower() for k in BOWLER_WICKET_KINDS])
    )

# Safe divide helper
def safe_div(num, den):
    return F.when(den.isNull() | (den == 0), F.lit(None)).otherwise(num / den)

# Rate helpers
def batting_strike_rate(runs_batter, balls_faced):
    return (safe_div(runs_batter * F.lit(100.0), balls_faced)).cast("double")

def bowling_economy(runs_conceded, balls_bowled):
    return (safe_div(runs_conceded * F.lit(6.0), balls_bowled)).cast("double")

def bowling_strike_rate(balls_bowled, wkts):
    return (safe_div(balls_bowled.cast("double"), wkts.cast("double"))).cast("double")