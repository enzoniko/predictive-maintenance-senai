"""Raw-file loading and schema validation for the case dataset.

Design notes (see docs/07_perguntas_ao_cliente.md for the open questions
these choices raise with the client):

* ``Classes.npy`` is a pickled ``dtype=object`` array of Python strings. Every
  ``np.load`` call on it therefore needs ``allow_pickle=True`` -- a supply
  -chain risk in production (a swapped file could execute arbitrary code on
  unpickling). ``load_labels`` converts it to a fixed-width ``<U`` string
  array immediately and callers never touch the pickled form again; the
  audit report flags this explicitly.
* ``Dados_1.npy``, ``Dados_2.npy`` and ``Dados_3.npy`` carry 201 columns
  instead of the expected 200 (10 kHz x 20 ms). Column 200 (0-indexed) is
  NaN in every row except exactly one, and that one non-NaN value has no
  consistent relationship to the row's class. It is dropped, not treated as
  a feature -- see ``_split_ghost_column`` for the validation that guards
  this assumption instead of silently trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pdm.config import Config

GHOST_COLUMN_MAX_NON_NAN_FRACTION = 0.01


class SchemaError(ValueError):
    """Raised when a raw file does not match the assumptions this loader encodes."""


@dataclass
class SensorDataset:
    """In-memory view of the case dataset: one (N, 200) matrix per sensor."""

    sensors: dict[str, np.ndarray]  # name -> (n_windows, window_len) float64
    labels: np.ndarray  # (n_windows,) fixed-width string array
    sample_rate_hz: int
    window_len: int
    ghost_columns: dict[str, np.ndarray]  # name -> raw 201st column, for the audit

    @property
    def n_windows(self) -> int:
        return self.labels.shape[0]

    @property
    def sensor_names(self) -> list[str]:
        return list(self.sensors.keys())

    def class_names(self) -> list[str]:
        return sorted(set(self.labels.tolist()))


def load_labels(labels_path: Path) -> np.ndarray:
    """Load Classes.npy and return a safe, non-pickled string array."""
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found. Run `python -m pdm.cli download-data` "
            "or scripts/download_data.py first."
        )
    raw = np.load(labels_path, allow_pickle=True)
    labels = np.asarray(raw, dtype=str).reshape(-1)
    return labels


def _split_ghost_column(
    name: str, matrix: np.ndarray, expected_len: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate and drop a trailing ghost column beyond ``expected_len``.

    Returns (clean_matrix, ghost_column_or_None). Raises SchemaError if the
    extra column does not match the "near-empty" pattern this loader was
    written to expect, so a genuinely different file shape fails loudly
    instead of silently losing a column of real data.
    """
    n_cols = matrix.shape[1]
    if n_cols == expected_len:
        return matrix, None
    if n_cols != expected_len + 1:
        raise SchemaError(
            f"{name}: expected {expected_len} or {expected_len + 1} columns, got {n_cols}."
        )
    ghost = matrix[:, expected_len]
    non_nan_fraction = np.mean(~np.isnan(ghost))
    if non_nan_fraction > GHOST_COLUMN_MAX_NON_NAN_FRACTION:
        raise SchemaError(
            f"{name}: column {expected_len} was expected to be a near-empty "
            f"artifact (<= {GHOST_COLUMN_MAX_NON_NAN_FRACTION:.0%} non-NaN) but "
            f"{non_nan_fraction:.1%} of rows are populated -- this may be real "
            "data, refusing to drop it silently."
        )
    return matrix[:, :expected_len], ghost


def load_sensor_dataset(config: Config) -> SensorDataset:
    """Load all configured sensor files plus labels into a SensorDataset."""
    labels = load_labels(config.paths.labels_path)
    window_len = config.acquisition["window_length_samples"]

    sensors: dict[str, np.ndarray] = {}
    ghost_columns: dict[str, np.ndarray] = {}
    for filename in config.paths.sensor_files:
        path = config.paths.sensor_path(filename)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m pdm.cli download-data` first."
            )
        matrix = np.load(path, mmap_mode="r")
        matrix = np.asarray(matrix, dtype=np.float64)
        clean, ghost = _split_ghost_column(filename, matrix, window_len)
        if clean.shape[0] != labels.shape[0]:
            raise SchemaError(
                f"{filename}: {clean.shape[0]} rows but Classes.npy has "
                f"{labels.shape[0]} labels."
            )
        name = Path(filename).stem
        sensors[name] = clean
        if ghost is not None:
            ghost_columns[name] = ghost

    return SensorDataset(
        sensors=sensors,
        labels=labels,
        sample_rate_hz=config.acquisition["sample_rate_hz"],
        window_len=window_len,
        ghost_columns=ghost_columns,
    )
