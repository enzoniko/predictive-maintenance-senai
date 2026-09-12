"""Configuration loading.

Every path in configs/default.yaml is relative to the repository root and is
resolved with pathlib, so the same config works unchanged on Windows and
Linux -- no hardcoded separators anywhere in this package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class Paths:
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    figures_dir: Path
    reports_dir: Path
    sensor_files: list[str]
    labels_file: str

    def sensor_path(self, filename: str) -> Path:
        return self.raw_dir / filename

    @property
    def labels_path(self) -> Path:
        return self.raw_dir / self.labels_file


@dataclass
class Config:
    paths: Paths
    acquisition: dict[str, Any]
    random_seed: int
    split: dict[str, Any]
    cross_validation: dict[str, Any]
    sensor_screening: dict[str, Any]
    cleaning: dict[str, Any]
    conformal: dict[str, Any]
    drift: dict[str, Any]
    imbalance: dict[str, Any]
    api: dict[str, Any]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def load_config(path: str | Path | None = None) -> Config:
    """Load configs/default.yaml (or an override) into a typed Config."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # PDM_DATABASE_URL overrides configs/default.yaml's api.database_url --
    # used by docker/docker-compose.yml to point the containerized API at
    # Postgres instead of the SQLite default, without a separate config file.
    env_database_url = os.environ.get("PDM_DATABASE_URL")
    if env_database_url:
        raw["api"]["database_url"] = env_database_url

    p = raw["paths"]
    paths = Paths(
        raw_dir=REPO_ROOT / p["raw_dir"],
        processed_dir=REPO_ROOT / p["processed_dir"],
        models_dir=REPO_ROOT / p["models_dir"],
        figures_dir=REPO_ROOT / p["figures_dir"],
        reports_dir=REPO_ROOT / p["reports_dir"],
        sensor_files=list(p["sensor_files"]),
        labels_file=p["labels_file"],
    )
    return Config(
        paths=paths,
        acquisition=raw["acquisition"],
        random_seed=raw["random_seed"],
        split=raw["split"],
        cross_validation=raw["cross_validation"],
        sensor_screening=raw["sensor_screening"],
        cleaning=raw["cleaning"],
        conformal=raw["conformal"],
        drift=raw["drift"],
        imbalance=raw["imbalance"],
        api=raw["api"],
        raw=raw,
    )
