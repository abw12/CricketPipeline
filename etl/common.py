from __future__ import annotations

import os
import shutil
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from pyspark.sql import DataFrame, SparkSession
from ruamel.yaml import YAML


def project_root(start: Optional[Path] = None) -> Path:
    env_root = os.getenv("CRICKET_PIPELINE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs").exists() and (candidate / "etl").exists():
            return candidate

    return Path(__file__).resolve().parents[1]


def load_yml(path_str: Union[str, Path]) -> dict[str, Any]:
    yaml = YAML(typ="safe")
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f) or {}


@dataclass(frozen=True)
class PipelineContext:
    root: Path
    bronze: dict[str, Any]
    silver: dict[str, Any]
    register: dict[str, Any]

    @classmethod
    def load(cls, root: Optional[Path] = None) -> "PipelineContext":
        base = project_root(root)
        return cls(
            root=base,
            bronze=load_yml(base / "configs" / "bronze_config.yml"),
            silver=load_yml(base / "configs" / "silver_config.yml"),
            register=load_yml(base / "configs" / "register_config.yml"),
        )

    def resolve(self, path: Union[str, Path]) -> Path:
        path_obj = Path(path)
        return path_obj if path_obj.is_absolute() else self.root / path_obj

    @property
    def bronze_format(self) -> str:
        return self.bronze["storage"]["format"]

    @property
    def silver_format(self) -> str:
        return self.silver["storage"]["format"]

    @property
    def gold_format(self) -> str:
        return self.silver["gold"]["format"]

    def bronze_table_path(self, table: str) -> Path:
        return self.resolve(self.bronze["tables"][table]["target_path"])

    def silver_table_path(self, table: str) -> Path:
        return self.resolve(self.silver["tables"][table]["target"])

    def gold_table_path(self, table: str) -> Path:
        return self.resolve(self.silver["gold"]["tables"][table])


def get_spark(app_name: str = "cricket-pipeline") -> SparkSession:
    root = project_root()
    event_dir = root / "logs" / "spark-events"
    event_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    return (
        SparkSession.builder.appName(app_name)  # type: ignore[attr-defined]
        .master("local[*]")
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.pyspark.python", sys.executable)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.hadoop.io.file.buffer.size", "65536")
        .config("spark.ui.port", "4040")
        .getOrCreate()
    )


def require_spark(
    app_name: str, spark: Optional[SparkSession] = None
) -> tuple[SparkSession, bool]:
    if spark is not None:
        return spark, False
    return get_spark(app_name), True


def read_table(spark: SparkSession, path: Path, fmt: str) -> DataFrame:
    return spark.read.format(fmt).load(str(path))


def write_table(
    df: DataFrame,
    target: Path,
    fmt: str,
    partition_columns: Iterable[str] = (),
    repartition_columns: Iterable[str] = (),
    mode: str = "overwrite",
    atomic: bool = False,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    write_target = _staging_path(target) if atomic and mode == "overwrite" else target

    out = df
    repartition_cols = list(repartition_columns)
    if repartition_cols:
        out = out.repartition(1, *repartition_cols)

    if write_target == target and mode == "overwrite" and target.exists():
        _remove_path(target)
        write_mode = "errorifexists"
    else:
        write_mode = mode

    writer = out.write.mode(write_mode).format(fmt)
    partitions = _active_partitions(partition_columns)
    if partitions:
        writer = writer.partitionBy(*partitions)
    writer.save(str(write_target))

    if write_target != target:
        _replace_path(write_target, target)
    return target


def show_preview(df: DataFrame, title: str, rows: int = 10) -> None:
    print(f"\n=== {title} ===")
    df.show(rows, truncate=False)
    df.printSchema()


def _staging_path(target: Path) -> Path:
    return target.with_name(f"_staging_{target.name}_{uuid.uuid4().hex}")


def _replace_path(staging: Path, target: Path) -> None:
    if target.exists():
        _remove_path(target)
    shutil.move(str(staging), str(target))


def _remove_path(path: Path) -> None:
    def make_writable(func, target, exc_info):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    shutil.rmtree(path, onerror=make_writable)


def _active_partitions(partition_columns: Iterable[str]) -> list[str]:
    if os.name == "nt" and os.getenv("CRICKET_PIPELINE_FORCE_PARTITIONS") != "1":
        return []
    return list(partition_columns)


