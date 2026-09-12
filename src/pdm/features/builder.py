"""Assembles the full feature table: cleaning -> time/frequency/wavelet features.

One row per window, one column group per retained sensor. Silent-window
flags are unioned across sensors into a single ``is_silent`` column so
downstream code (evaluation/conformal.py, serving/api.py) has one place to
decide how to handle a window nothing was confidently measured on.
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


def build_feature_table(
    dataset: SensorDataset,
    config: Config,
    sensors: list[str] | None = None,
    include_wavelet: bool = True,
) -> pd.DataFrame:
    """Build the engineered feature table for the given (or all) sensors.

    ``sensors`` is normally the audit's ``recommended_sensors`` -- passing an
    explicit list keeps this function decoupled from the audit, so a
    notebook can still ask "what if I fed it Dados_4 too?" without editing
    this module.
    """
    sensor_names = sensors if sensors is not None else dataset.sensor_names
    cleaning_cfg = config.cleaning
    fs = dataset.sample_rate_hz

    feature_frames: list[pd.DataFrame] = []
    silent_masks: list[np.ndarray] = []

    for name in sensor_names:
        raw = dataset.sensors[name]
        cleaner = SensorCleaner(
            saturation_abs_threshold=cleaning_cfg["saturation_abs_threshold"],
            hampel_window=cleaning_cfg["hampel_window"],
            hampel_n_sigmas=cleaning_cfg["hampel_n_sigmas"],
            silent_window_std_percentile=cleaning_cfg["silent_window_std_percentile"],
        )
        cleaned = cleaner.fit_transform(raw)
        silent_masks.append(cleaner.silent_mask_)

        feature_frames.append(time_domain_features(cleaned, prefix=name))
        feature_frames.append(frequency_domain_features(cleaned, prefix=name, fs=fs))
        if include_wavelet:
            feature_frames.append(wavelet_features(cleaned, prefix=name, fs=fs))

    table = pd.concat(feature_frames, axis=1)
    table["is_silent"] = np.any(np.stack(silent_masks, axis=1), axis=1)
    table["label"] = dataset.labels
    return table
