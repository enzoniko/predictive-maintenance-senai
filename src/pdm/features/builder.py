"""Assembles the full feature table: cleaning -> time/frequency/wavelet features.

Two entry points, deliberately kept separate:

* ``build_feature_table`` fits cleaning thresholds (saturation, silence) and
  extracts features from the *same* data in one step. Correct for
  exploratory analysis over the whole dataset (notebooks 01/02) and for the
  audit, where there is no train/test boundary to protect.
* ``fit_cleaners`` + ``build_features_from_raw`` split that into a fit step
  (thresholds learned from training rows only) and a transform step
  (applied to train/calibration/test independently). ``models/pipeline.py``
  uses this pair -- fitting cleaning thresholds on the full dataset before
  splitting would leak calibration/test information into a value baked
  directly into the features the model trains on, which is exactly the kind
  of blind trust this project's data-audit ethos (see
  docs/01_interpretacao_problema.md) argues against.

One row per window, one column group per sensor. Silent-window flags are
unioned across sensors into a single ``is_silent`` column so downstream code
(evaluation/conformal.py, serving/api.py) has one place to decide how to
handle a window nothing was confidently measured on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.config import Config
from pdm.data.loader import SensorDataset
from pdm.features.frequency_domain import frequency_domain_features
from pdm.features.time_domain import time_domain_features
from pdm.features.wavelet import wavelet_features
from pdm.preprocessing.cleaning import SensorCleaner


def _make_cleaner(config: Config) -> SensorCleaner:
    c = config.cleaning
    return SensorCleaner(
        saturation_abs_threshold=c["saturation_abs_threshold"],
        hampel_window=c["hampel_window"],
        hampel_n_sigmas=c["hampel_n_sigmas"],
        silent_window_std_percentile=c["silent_window_std_percentile"],
    )


def fit_cleaners(
    dataset: SensorDataset, config: Config, sensors: list[str], row_indices: np.ndarray | None = None
) -> dict[str, SensorCleaner]:
    """Fit one SensorCleaner per sensor, on ``row_indices`` only if given
    (pass the training split's indices to avoid leaking calibration/test
    rows into the fitted thresholds)."""
    cleaners = {}
    for name in sensors:
        raw = dataset.sensors[name]
        if row_indices is not None:
            raw = raw[row_indices]
        cleaners[name] = _make_cleaner(config).fit(raw)
    return cleaners


def build_features_from_raw(
    sensors_raw: dict[str, np.ndarray],
    cleaners: dict[str, SensorCleaner],
    fs: int,
    include_wavelet: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply already-fitted cleaners and extract features. Returns
    ``(feature_table_without_label, is_silent_mask)`` -- the caller attaches
    whatever label array corresponds to these rows."""
    feature_frames: list[pd.DataFrame] = []
    silent_masks: list[np.ndarray] = []

    for name, cleaner in cleaners.items():
        cleaned = cleaner.transform(sensors_raw[name])
        silent_masks.append(cleaner.silent_mask_)
        feature_frames.append(time_domain_features(cleaned, prefix=name))
        feature_frames.append(frequency_domain_features(cleaned, prefix=name, fs=fs))
        if include_wavelet:
            feature_frames.append(wavelet_features(cleaned, prefix=name, fs=fs))

    table = pd.concat(feature_frames, axis=1)
    is_silent = np.any(np.stack(silent_masks, axis=1), axis=1)
    return table, is_silent


def build_feature_table(
    dataset: SensorDataset,
    config: Config,
    sensors: list[str] | None = None,
    include_wavelet: bool = True,
) -> pd.DataFrame:
    """Fit cleaners and build features from the same (whole) dataset -- see
    the module docstring for when this is (and is not) the right choice.

    ``sensors`` is normally the audit's ``recommended_sensors`` -- passing an
    explicit list keeps this function decoupled from the audit, so a
    notebook can still ask "what if I fed it Dados_4 too?" without editing
    this module.
    """
    sensor_names = sensors if sensors is not None else dataset.sensor_names
    cleaners = fit_cleaners(dataset, config, sensor_names)
    raw = {name: dataset.sensors[name] for name in sensor_names}
    table, is_silent = build_features_from_raw(raw, cleaners, dataset.sample_rate_hz, include_wavelet)
    table["is_silent"] = is_silent
    table["label"] = dataset.labels
    return table
