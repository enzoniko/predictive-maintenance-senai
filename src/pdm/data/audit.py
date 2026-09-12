"""Automated data-quality audit.

This module is the codified version of the skepticism a project of this kind
demands: instead of trusting the case statement's claims (10 kHz, simultaneous
acquisition, "different failure states"), it re-derives what can be verified
from the arrays themselves and flags what cannot. See
docs/07_perguntas_ao_cliente.md for the questions this audit could not
resolve on its own, and docs/01_interpretacao_problema.md for how the
findings below shaped the project's scope.

Every check returns a small dataclass with a human-readable ``verdict`` plus
the numbers behind it, so the CLI/report can render this as a table without
re-deriving anything, and a unit test can assert on the verdict alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from pdm.data.loader import SensorDataset

# A channel whose lag-1 autocorrelation sits within this band of zero, on
# every class, is indistinguishable from white noise for our purposes.
WHITENESS_AUTOCORR_MAX = 0.15
# A channel is "stuck" if almost all samples take on a single value.
STUCK_UNIQUE_RATIO_MAX = 0.001
SILENT_WINDOW_STD_PERCENTILE = 2.0


@dataclass
class CheckResult:
    name: str
    verdict: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SensorAudit:
    sensor: str
    n_nan: int
    nan_fraction: float
    saturation_hits: int
    saturation_threshold: float
    silent_windows: int
    silent_window_fraction: float
    is_stuck: bool
    unique_value_ratio: float
    is_white_noise: bool
    lag1_autocorr_by_class: dict[str, float]
    kruskal_wallis_p: float
    mutual_info_with_label: float
    permutation_test_p: float
    verdict: str


@dataclass
class AuditReport:
    n_windows: int
    window_len: int
    sample_rate_hz: int
    class_balance: dict[str, int]
    class_balance_verdict: str
    ghost_columns: dict[str, CheckResult]
    row_contiguity: CheckResult
    inter_sensor_simultaneity: CheckResult
    sensor_audits: dict[str, SensorAudit]
    recommended_sensors: list[str]
    excluded_sensors: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_windows": self.n_windows,
            "window_len": self.window_len,
            "sample_rate_hz": self.sample_rate_hz,
            "class_balance": self.class_balance,
            "class_balance_verdict": self.class_balance_verdict,
            "ghost_columns": {k: v.to_dict() for k, v in self.ghost_columns.items()},
            "row_contiguity": self.row_contiguity.to_dict(),
            "inter_sensor_simultaneity": self.inter_sensor_simultaneity.to_dict(),
            "sensor_audits": {k: asdict(v) for k, v in self.sensor_audits.items()},
            "recommended_sensors": self.recommended_sensors,
            "excluded_sensors": self.excluded_sensors,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)


def _check_ghost_columns(dataset: SensorDataset) -> dict[str, CheckResult]:
    results = {}
    for name, ghost in dataset.ghost_columns.items():
        non_nan_idx = np.where(~np.isnan(ghost))[0]
        results[name] = CheckResult(
            name=f"ghost_column[{name}]",
            verdict=(
                "empty artifact column, dropped"
                if len(non_nan_idx) <= 1
                else "unexpected: multiple non-NaN entries, review manually"
            ),
            details={
                "n_non_nan": int(len(non_nan_idx)),
                "non_nan_row_indices": non_nan_idx[:20].tolist(),
                "non_nan_values": ghost[non_nan_idx][:20].tolist(),
                "hypotheses": [
                    "end-of-window delimiter -- rejected: not constant across rows",
                    "leftover label/index column misaligned during export",
                    "artifact of concatenating a (N, 201) DataFrame with a stray column",
                ],
            },
        )
    return results


def _check_row_contiguity(dataset: SensorDataset) -> CheckResult:
    """Rows are supposed to be independent windows; verify they are not a
    shuffled view of one continuous recording (which would change how CV
    splits must be built to avoid leakage)."""
    name = next(iter(dataset.sensors))
    a = dataset.sensors[name]
    within = np.abs(np.diff(a, axis=1)).mean()
    between = np.abs(a[1:, 0] - a[:-1, -1]).mean()
    ratio = between / within if within > 0 else float("nan")
    contiguous = ratio < 1.3
    return CheckResult(
        name="row_contiguity",
        verdict=(
            "rows look contiguous (continuous recording sliced into windows) "
            "-- group-aware splitting required"
            if contiguous
            else "rows are independent/shuffled windows -- no evidence of a "
            "continuous underlying recording"
        ),
        details={
            "reference_sensor": name,
            "mean_abs_diff_within_row": float(within),
            "mean_abs_diff_between_consecutive_rows": float(between),
            "ratio": float(ratio),
        },
    )


def _check_simultaneity(dataset: SensorDataset) -> CheckResult:
    """No timestamps are provided, so exact simultaneity cannot be proven.
    As a proxy, check whether the real (non-noise) sensors co-vary in
    amplitude row-by-row, which is what one would expect if they observed the
    same operating condition at the same time."""
    real_sensors = [n for n in dataset.sensor_names if n not in ("Dados_4", "Dados_5")]
    corrs = {}
    for i, a in enumerate(real_sensors):
        for b in real_sensors[i + 1 :]:
            rms_a = dataset.sensors[a].std(axis=1)
            rms_b = dataset.sensors[b].std(axis=1)
            corrs[f"{a}-{b}"] = float(np.corrcoef(rms_a, rms_b)[0, 1])
    mean_corr = float(np.mean(list(corrs.values()))) if corrs else float("nan")
    return CheckResult(
        name="inter_sensor_simultaneity",
        verdict=(
            "cannot verify exact timestamps (none provided); row-wise RMS "
            "envelopes co-vary across sensors, consistent with -- but not "
            "proof of -- simultaneous acquisition"
            if mean_corr > 0.4
            else "sensors do not visibly co-vary; the 'simultaneous acquisition' "
            "claim could not be corroborated"
        ),
        details={"row_rms_correlations": corrs, "mean_correlation": mean_corr},
    )


def _class_balance(dataset: SensorDataset) -> tuple[dict[str, int], str]:
    classes, counts = np.unique(dataset.labels, return_counts=True)
    balance = {c: int(n) for c, n in zip(classes, counts)}
    spread = (max(counts) - min(counts)) / max(counts)
    verdict = (
        "perfectly balanced across classes -- unlikely to occur naturally on "
        "a factory floor; treat as a curated/synthetic sample and re-evaluate "
        "under a realistic class prior before deployment (see "
        "evaluation/imbalance.py)"
        if spread < 0.01
        else f"imbalanced (max-min spread {spread:.1%})"
    )
    return balance, verdict


def _mutual_info_with_label(values: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform(labels)
    mi = mutual_info_classif(
        values.reshape(-1, 1), y, discrete_features=False, random_state=0, n_neighbors=3
    )
    return float(mi[0])


def _permutation_control_test(
    values: np.ndarray,
    labels: np.ndarray,
    n_repeats: int,
    seed: int,
    max_samples: int = 5000,
) -> float:
    """A control akin to a permutation test: shuffle labels and see how often
    a real-looking mutual-information score arises by chance for this
    specific (possibly degenerate) channel.

    Subsamples to ``max_samples`` rows (consistently for the observed score
    and every null draw) purely for runtime -- 20k+ rows would make the
    k-NN-based MI estimator the bottleneck of the whole audit for no gain in
    the statistic's reliability.

    Resolution caveat: with ``n_repeats`` draws the smallest achievable
    p-value is ``1 / (n_repeats + 1)``, so ``n_repeats`` must be chosen well
    below the significance threshold (``sensor_screening.alpha``) or even a
    strongly informative channel could never clear the bar. The default
    config uses 199 repeats for alpha=0.01 to leave headroom.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        values = values[idx]
        labels = labels[idx]
    observed = _mutual_info_with_label(values, labels)
    if observed == 0.0:
        return 1.0
    null = np.empty(n_repeats)
    for i in range(n_repeats):
        shuffled = rng.permutation(labels)
        null[i] = _mutual_info_with_label(values, shuffled)
    p_value = float((np.sum(null >= observed) + 1) / (n_repeats + 1))
    return p_value


def _audit_sensor(
    name: str,
    matrix: np.ndarray,
    labels: np.ndarray,
    saturation_threshold: float,
    n_perm_repeats: int,
    alpha: float,
    seed: int,
) -> SensorAudit:
    nan_mask = np.isnan(matrix)
    n_nan = int(nan_mask.sum())
    nan_fraction = float(nan_mask.mean())

    filled = np.where(nan_mask, np.nanmean(matrix), matrix)
    saturation_hits = int(np.sum(np.abs(filled) >= saturation_threshold))

    row_std = filled.std(axis=1)
    silent_threshold = np.percentile(row_std, SILENT_WINDOW_STD_PERCENTILE)
    silent_windows = int(np.sum(row_std <= silent_threshold))

    unique_ratio = len(np.unique(np.round(filled, 6))) / filled.size
    is_stuck = unique_ratio < STUCK_UNIQUE_RATIO_MAX

    classes = sorted(set(labels.tolist()))
    lag1_by_class: dict[str, float] = {}
    for c in classes:
        rows = filled[labels == c]
        centered = rows - rows.mean(axis=1, keepdims=True)
        num = np.sum(centered[:, :-1] * centered[:, 1:])
        den = np.sum(centered[:, :-1] ** 2) + 1e-12
        lag1_by_class[c] = float(num / den)
    is_white_noise = all(abs(v) < WHITENESS_AUTOCORR_MAX for v in lag1_by_class.values())

    # RMS (row_std) rather than the row mean is used as the univariate
    # class-association probe: for vibration/current-like signals the class
    # information lives in signal *energy* and spectral shape, not in the DC
    # level. Using the row mean instead gave a borderline, resolution-limited
    # p-value for Dados_1 despite that channel alone supporting F1 > 0.75 in
    # a downstream classifier (see notebooks/01_data_audit.ipynb). This
    # univariate probe is deliberately coarse and only meant to catch
    # dead/noise channels early; the real feature set built in features/ is
    # far richer and is what modelling decisions in models/ are based on.
    groups = [row_std[labels == c] for c in classes]
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            _, kw_p = stats.kruskal(*groups)
        # A (near-)constant channel ties every observation, and the H-statistic's
        # tie correction divides by zero -- kruskal then returns nan rather
        # than raising. Treat that degenerate case as "no evidence".
        if np.isnan(kw_p):
            kw_p = 1.0
    except ValueError:
        kw_p = 1.0

    mi = _mutual_info_with_label(row_std, labels)
    perm_p = _permutation_control_test(row_std, labels, n_perm_repeats, seed)

    carries_info = (kw_p < alpha) and (perm_p < alpha) and not is_stuck
    if is_stuck:
        verdict = "stuck at (near-)constant value -- no information, likely a dead/miswired sensor"
    elif is_white_noise and not carries_info:
        verdict = "statistically indistinguishable from white noise across classes -- excluded"
    elif carries_info:
        verdict = "carries class-discriminative information -- retained"
    else:
        verdict = "no significant class association detected at the tested resolution -- excluded"

    return SensorAudit(
        sensor=name,
        n_nan=n_nan,
        nan_fraction=nan_fraction,
        saturation_hits=saturation_hits,
        saturation_threshold=saturation_threshold,
        silent_windows=silent_windows,
        silent_window_fraction=silent_windows / len(row_std),
        is_stuck=is_stuck,
        unique_value_ratio=unique_ratio,
        is_white_noise=is_white_noise,
        lag1_autocorr_by_class=lag1_by_class,
        kruskal_wallis_p=float(kw_p),
        mutual_info_with_label=mi,
        permutation_test_p=perm_p,
        verdict=verdict,
    )


def run_audit(dataset: SensorDataset, config) -> AuditReport:  # noqa: ANN001 (Config, avoids import cycle)
    balance, balance_verdict = _class_balance(dataset)
    ghost = _check_ghost_columns(dataset)
    contiguity = _check_row_contiguity(dataset)
    simultaneity = _check_simultaneity(dataset)

    sc = config.sensor_screening
    sensor_audits: dict[str, SensorAudit] = {}
    for name, matrix in dataset.sensors.items():
        sensor_audits[name] = _audit_sensor(
            name,
            matrix,
            dataset.labels,
            saturation_threshold=config.cleaning["saturation_abs_threshold"],
            n_perm_repeats=sc["permutation_n_repeats"],
            alpha=sc["alpha"],
            seed=config.random_seed,
        )

    recommended = [n for n, a in sensor_audits.items() if "retained" in a.verdict]
    excluded = {n: a.verdict for n, a in sensor_audits.items() if "retained" not in a.verdict}

    return AuditReport(
        n_windows=dataset.n_windows,
        window_len=dataset.window_len,
        sample_rate_hz=dataset.sample_rate_hz,
        class_balance=balance,
        class_balance_verdict=balance_verdict,
        ghost_columns=ghost,
        row_contiguity=contiguity,
        inter_sensor_simultaneity=simultaneity,
        sensor_audits=sensor_audits,
        recommended_sensors=recommended,
        excluded_sensors=excluded,
    )
