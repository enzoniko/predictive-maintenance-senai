"""Feature-table caching, with a Parquet-or-joblib fallback.

``DataFrame.to_parquet``/``read_parquet`` need ``pyarrow`` (or
``fastparquet``), a compiled dependency that not every environment builds
cleanly; the same root cause already documented for
mlflow/shap/ssqueezepy (see docs/03_arquitetura.md, section 3.6). Parquet
is still tried first (smaller files, preserves dtypes precisely, and is
what the deployment target uses), but
``save_feature_table``/``load_feature_table`` transparently fall back to a
joblib-pickled DataFrame; same API, same call sites, no
environment-specific branching needed outside this module.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

PARQUET_FILENAME = "features.parquet"
FALLBACK_FILENAME = "features.joblib"


def save_feature_table(table: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / PARQUET_FILENAME
    try:
        table.to_parquet(parquet_path)
        return parquet_path
    except ImportError:
        fallback_path = out_dir / FALLBACK_FILENAME
        joblib.dump(table, fallback_path)
        return fallback_path


def load_feature_table(out_dir: Path) -> pd.DataFrame:
    parquet_path = out_dir / PARQUET_FILENAME
    fallback_path = out_dir / FALLBACK_FILENAME
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except ImportError:
            pass
    if fallback_path.exists():
        return joblib.load(fallback_path)
    raise FileNotFoundError(
        f"No cached feature table in {out_dir}; run `python -m pdm.cli features` first."
    )
