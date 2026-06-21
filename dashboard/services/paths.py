from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    env_root = os.getenv("CRICKET_PIPELINE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DashboardPaths:
    root: Path

    @classmethod
    def load(cls) -> "DashboardPaths":
        return cls(root=project_root())

    @property
    def gold_dir(self) -> Path:
        return self.root / "data" / "processed" / "gold"

    @property
    def silver_dir(self) -> Path:
        return self.root / "data" / "processed" / "silver"

    def gold_table(self, table_name: str) -> Path:
        return self.gold_dir / table_name

    def silver_table(self, table_name: str) -> Path:
        return self.silver_dir / table_name
