"""Feature-distribution drift monitoring (PSI + Kolmogorov-Smirnov).

Fit once on a reference distribution (the training set), then score any
later batch -- one incoming production batch, or a whole day/week of
predictions pulled back from the database (see serving/db.py) -- for how
far each feature has moved. This is what turns "the model was 96% accurate
in the case study" into an operational claim: without drift monitoring, a
model silently degrading as the machine wears, gets serviced, or has its
operating envelope changed would look identical to a healthy model from the
outside, until it is confidently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

PSI_EPS = 1e-6


def _psi_for_feature(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two 1-D samples, using
    reference-quantile bins so each reference bin starts with ~equal mass."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        return 0.0  # degenerate (near-constant) feature -- nothing to compare
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_frac = ref_counts / max(len(reference), 1) + PSI_EPS
    cur_frac = cur_counts / max(len(current), 1) + PSI_EPS
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    alarm: bool


@dataclass
class DriftReport:
    n_reference: int
    n_current: int
    features: list[FeatureDriftResult] = field(default_factory=list)

    @property
    def n_alarms(self) -> int:
        return sum(1 for f in self.features if f.alarm)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.features]).sort_values(
            "psi", ascending=False
        )


class DriftMonitor:
    def __init__(self, psi_alarm_threshold: float = 0.2, n_bins: int = 10) -> None:
        self.psi_alarm_threshold = psi_alarm_threshold
        self.n_bins = n_bins
        self.reference_: pd.DataFrame | None = None

    def fit(self, reference: pd.DataFrame) -> DriftMonitor:
        self.reference_ = reference.copy()
        return self

    def score(self, current: pd.DataFrame) -> DriftReport:
        if self.reference_ is None:
            raise RuntimeError("call fit() before score()")
        common_cols = [c for c in self.reference_.columns if c in current.columns]
        results = []
        for col in common_cols:
            ref_vals = self.reference_[col].dropna().to_numpy()
            cur_vals = current[col].dropna().to_numpy()
            if len(ref_vals) < 2 or len(cur_vals) < 2:
                continue
            psi = _psi_for_feature(ref_vals, cur_vals, self.n_bins)
            ks_stat, ks_p = stats.ks_2samp(ref_vals, cur_vals)
            results.append(
                FeatureDriftResult(
                    feature=col,
                    psi=psi,
                    ks_statistic=float(ks_stat),
                    ks_p_value=float(ks_p),
                    alarm=psi >= self.psi_alarm_threshold,
                )
            )
        return DriftReport(n_reference=len(self.reference_), n_current=len(current), features=results)
