"""Shared fixtures: a small synthetic dataset mirroring the case's schema.

Tests never touch the real ~400 MB Dados_*.npy files (those are gitignored
and only present locally after scripts/download_data.py runs). Instead this
builds a tiny synthetic dataset with the same shape quirks; a 201st ghost
column on the "real" sensors, one stuck sensor, one white-noise sensor; so
the loader and audit logic are exercised end to end quickly and
deterministically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pdm.config import Config, Paths

N_WINDOWS = 600
WINDOW_LEN = 200
CLASSES = ["Classe A", "Classe B", "Classe C", "Classe D", "Classe E"]


def _make_labels(rng: np.random.Generator) -> np.ndarray:
    """Balanced labels (N_WINDOWS must be divisible by len(CLASSES)), shuffled."""
    assert N_WINDOWS % len(CLASSES) == 0
    labels = np.repeat(CLASSES, N_WINDOWS // len(CLASSES))
    rng.shuffle(labels)
    return labels


def _make_real_sensor(rng: np.random.Generator, labels: np.ndarray, scale: float) -> np.ndarray:
    """A sensor whose window-to-window RMS depends on the class label."""
    class_scale = {c: scale * (1 + 0.5 * i) for i, c in enumerate(CLASSES)}
    rows = [rng.normal(0, class_scale[label], size=WINDOW_LEN) for label in labels]
    return np.asarray(rows)


def _add_ghost_column(matrix: np.ndarray, rng: np.random.Generator, marker_value: float) -> np.ndarray:
    ghost = np.full((matrix.shape[0], 1), np.nan)
    ghost[rng.integers(0, matrix.shape[0])] = marker_value
    return np.hstack([matrix, ghost])


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    labels = _make_labels(rng)
    np.save(raw_dir / "Classes.npy", labels.reshape(-1, 1).astype(object), allow_pickle=True)

    dados_1 = _add_ghost_column(_make_real_sensor(rng, labels, scale=0.1), rng, marker_value=2.0)
    dados_2 = _add_ghost_column(_make_real_sensor(rng, labels, scale=0.2), rng, marker_value=4.0)
    dados_3 = _add_ghost_column(_make_real_sensor(rng, labels, scale=0.3), rng, marker_value=2.0)
    dados_4 = np.full((N_WINDOWS, WINDOW_LEN), 50.0)  # stuck sensor
    dados_5 = rng.uniform(-20, 140, size=(N_WINDOWS, WINDOW_LEN))  # white noise, no class link

    np.save(raw_dir / "Dados_1.npy", dados_1)
    np.save(raw_dir / "Dados_2.npy", dados_2)
    np.save(raw_dir / "Dados_3.npy", dados_3)
    np.save(raw_dir / "Dados_4.npy", dados_4)
    np.save(raw_dir / "Dados_5.npy", dados_5)
    return raw_dir


@pytest.fixture
def synthetic_config(synthetic_raw_dir: Path, tmp_path: Path) -> Config:
    paths = Paths(
        raw_dir=synthetic_raw_dir,
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
        figures_dir=tmp_path / "figures",
        reports_dir=tmp_path / "reports",
        sensor_files=["Dados_1.npy", "Dados_2.npy", "Dados_3.npy", "Dados_4.npy", "Dados_5.npy"],
        labels_file="Classes.npy",
    )
    return Config(
        paths=paths,
        acquisition={"sample_rate_hz": 10000, "window_length_samples": WINDOW_LEN},
        random_seed=0,
        split={"train": 0.8, "calibration": 0.1, "test": 0.1, "stratify": True},
        cross_validation={"n_splits": 3, "shuffle": True},
        sensor_screening={
            "min_unique_ratio": 0.05,
            # Looser than production (0.01) because this fixture uses far
            # fewer permutation repeats for speed; see the note below.
            "alpha": 0.05,
            "whiteness_autocorr_lag1_max": 0.15,
            # Fewer repeats than production for fast, deterministic tests;
            # floor is 1/(n+1) = 0.02, comfortably below alpha=0.05 above.
            "permutation_n_repeats": 49,
        },
        cleaning={
            "saturation_abs_threshold": 5.0,
            "hampel_window": 7,
            "hampel_n_sigmas": 3.0,
            "silent_window_std_percentile": 2.0,
        },
        conformal={"method": "aps", "target_coverage": 0.90},
        drift={"method": "psi", "psi_alarm_threshold": 0.2},
        imbalance={"simulated_prior": {c: 0.2 for c in CLASSES}},
        api={"host": "0.0.0.0", "port": 8000, "database_url": "sqlite:///:memory:"},
    )
