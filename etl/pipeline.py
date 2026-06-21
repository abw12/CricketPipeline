from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from pyspark.sql import SparkSession

from etl.common import PipelineContext, get_spark


StepRunner = Callable[..., None]


@dataclass(frozen=True)
class PipelineStep:
    name: str
    stage: str
    runner: StepRunner


def _steps() -> tuple[PipelineStep, ...]:
    from etl.bronze_ingest.write_deliveries import run as bronze_deliveries
    from etl.bronze_ingest.write_innings import run as bronze_innings
    from etl.bronze_ingest.write_matches import run as bronze_matches
    from etl.gold.build_batter_season import run as gold_batter_season
    from etl.gold.build_bowler_season import run as gold_bowler_season
    from etl.gold.build_match_summary import run as gold_match_summary
    from etl.gold.build_team_phase_season import run as gold_team_phase_season
    from etl.silver_transform.build_dim_player import run as silver_dim_player
    from etl.silver_transform.build_dim_team import run as silver_dim_team
    from etl.silver_transform.enrich_deliveries_players import (
        run as silver_enrich_deliveries_players,
    )
    from etl.silver_transform.enrich_deliveries_teams import (
        run as silver_enrich_deliveries_teams,
    )
    from etl.silver_transform.enrich_matches_players import (
        run as silver_enrich_matches_players,
    )
    from etl.silver_transform.enrich_matches_teams import (
        run as silver_enrich_matches_teams,
    )
    from etl.silver_transform.normalize_deliveries import (
        run as silver_normalize_deliveries,
    )
    from etl.silver_transform.normalize_matches import run as silver_normalize_matches

    return (
        PipelineStep("bronze-matches", "bronze", bronze_matches),
        PipelineStep("bronze-deliveries", "bronze", bronze_deliveries),
        PipelineStep("bronze-innings", "bronze", bronze_innings),
        PipelineStep("silver-matches", "silver", silver_normalize_matches),
        PipelineStep("silver-deliveries", "silver", silver_normalize_deliveries),
        PipelineStep("silver-dim-player", "silver", silver_dim_player),
        PipelineStep("silver-dim-team", "silver", silver_dim_team),
        PipelineStep(
            "silver-deliveries-player-ids",
            "silver",
            silver_enrich_deliveries_players,
        ),
        PipelineStep(
            "silver-deliveries-team-ids",
            "silver",
            silver_enrich_deliveries_teams,
        ),
        PipelineStep("silver-matches-player-ids", "silver", silver_enrich_matches_players),
        PipelineStep("silver-matches-team-ids", "silver", silver_enrich_matches_teams),
        PipelineStep("gold-batter-season", "gold", gold_batter_season),
        PipelineStep("gold-bowler-season", "gold", gold_bowler_season),
        PipelineStep("gold-team-phase-season", "gold", gold_team_phase_season),
        PipelineStep("gold-match-summary", "gold", gold_match_summary),
    )


def available_steps() -> tuple[PipelineStep, ...]:
    return _steps()


def run_pipeline(
    stage: str = "all",
    step_name: Optional[str] = None,
    sample_only: bool = False,
    write_out: bool = True,
    preview: bool = False,
) -> None:
    context = PipelineContext.load()
    spark = get_spark(f"cricket-pipeline-{step_name or stage}")
    try:
        for step in _selected_steps(stage, step_name):
            print(f"\n>>> Running {step.name}")
            step.runner(
                context=context,
                spark=spark,
                sample_only=sample_only,
                write_out=write_out,
                preview=preview,
            )
    finally:
        spark.stop()


def run_step(
    runner: StepRunner,
    context: PipelineContext,
    spark: SparkSession,
    sample_only: bool,
    write_out: bool,
    preview: bool,
) -> None:
    runner(
        context=context,
        spark=spark,
        sample_only=sample_only,
        write_out=write_out,
        preview=preview,
    )


def _selected_steps(stage: str, step_name: Optional[str]) -> tuple[PipelineStep, ...]:
    steps = _steps()
    if step_name:
        matches = tuple(step for step in steps if step.name == step_name)
        if not matches:
            valid = ", ".join(step.name for step in steps)
            raise ValueError(f"Unknown step '{step_name}'. Valid steps: {valid}")
        return matches

    if stage == "all":
        return steps

    matches = tuple(step for step in steps if step.stage == stage)
    if not matches:
        raise ValueError("Stage must be one of: bronze, silver, gold, all")
    return matches


