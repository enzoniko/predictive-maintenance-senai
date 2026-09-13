"""Split conformal prediction: prediction *sets* with a distribution-free
coverage guarantee, instead of a single point prediction the model always
commits to.

This is the concrete answer to "how does the system know when it does not
know?"; for a maintenance team, a prediction set of size 1 with high
confidence is actionable on its own, a set of size 3-4 says "narrow it down
with a manual inspection", and (with LAC) an *empty* set says "this window
does not look like anything the model has been trained on, escalate it".

Two scoring rules are implemented:

* **LAC** (least ambiguous classifier): nonconformity = 1; P(true class).
  Simple, can produce small or empty sets, but set sizes vary more sharply
  with how "peaked" a prediction is.
* **APS** (adaptive prediction sets, Romano et al. 2020): nonconformity =
  cumulative probability mass of all classes at least as likely as the true
  one. Its coverage is more stable across confidence regimes than LAC's,
  which is why it is the default in configs/default.yaml.

APS is implemented in its **randomized** form. The plain (deterministic)
version; always fully including the class at which the cumulative sum
crosses the threshold; is a well-documented source of systematic
over-coverage (see Romano et al. 2020, Angelopoulos & Bates's conformal
tutorial): with few classes in particular, that boundary class alone can
carry a large slice of probability mass, so "always include it" can inflate
empirical coverage well past the target instead of tracking it. The
randomized version includes that boundary class only with probability
proportional to how much of the threshold its own mass would fill, using a
per-sample uniform draw; this is what actually delivers the target
coverage rather than a loose upper bound on it. Reproducibility is
controlled by ``random_state``, threaded through both calibration and
prediction.

Both require a held-out **calibration** split, disjoint from both training
and the reported test set (see the 80/10/10 split in configs/default.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _class_index_map(classes: list[str]) -> dict[str, int]:
    return {c: i for i, c in enumerate(classes)}


def _lac_scores(proba: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    return 1.0 - proba[np.arange(len(y_idx)), y_idx]


def _aps_scores(proba: np.ndarray, y_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(y_idx)
    order = np.argsort(-proba, axis=1)
    sorted_proba = np.take_along_axis(proba, order, axis=1)
    cumsum = np.cumsum(sorted_proba, axis=1)
    rank = np.argmax(order == y_idx[:, None], axis=1)

    true_proba = proba[np.arange(n), y_idx]
    cumsum_before = cumsum[np.arange(n), rank] - true_proba
    u = rng.uniform(size=n)
    return cumsum_before + u * true_proba


def _lac_sets(proba: np.ndarray, q_hat: float) -> np.ndarray:
    return proba >= (1.0 - q_hat)


def _aps_sets(proba: np.ndarray, q_hat: float, rng: np.random.Generator) -> np.ndarray:
    n, k = proba.shape
    order = np.argsort(-proba, axis=1)
    sorted_proba = np.take_along_axis(proba, order, axis=1)
    cumsum = np.cumsum(sorted_proba, axis=1)
    reaches = cumsum >= q_hat
    # If no rank reaches q_hat (only possible if q_hat > total prob. mass,
    # i.e. q_hat > ~1), the boundary is the last rank and gets included
    # outright below (frac clipped to 1).
    any_reach = reaches.any(axis=1)
    boundary = np.where(any_reach, np.argmax(reaches, axis=1), k - 1)

    idx = np.arange(n)
    boundary_proba = sorted_proba[idx, boundary]
    cumsum_before_boundary = cumsum[idx, boundary] - boundary_proba
    frac = np.clip((q_hat - cumsum_before_boundary) / np.clip(boundary_proba, 1e-12, None), 0.0, 1.0)
    include_boundary = rng.uniform(size=n) <= frac
    include_upto = boundary - 1 + include_boundary.astype(int)

    rank_grid = np.arange(k)[None, :]
    include_mask_sorted = rank_grid <= include_upto[:, None]  # rows with include_upto=-1 -> all False
    sets = np.zeros((n, k), dtype=bool)
    np.put_along_axis(sets, order, include_mask_sorted, axis=1)
    return sets


@dataclass
class ConformalCalibrationSummary:
    method: str
    target_coverage: float
    q_hat: float
    n_calibration: int


class SplitConformalClassifier:
    def __init__(self, classes: list[str], method: str = "aps", random_state: int = 42) -> None:
        if method not in ("aps", "lac"):
            raise ValueError(f"unknown method {method!r}, expected 'aps' or 'lac'")
        self.classes = classes
        self.method = method
        self.class_index = _class_index_map(classes)
        self.q_hat_: float | None = None
        # Only APS consumes randomness (see module docstring); a fixed seed
        # keeps predict_sets() reproducible across calls and processes.
        self._calib_rng = np.random.default_rng(random_state)
        self._predict_rng = np.random.default_rng(random_state + 1)

    def calibrate(
        self, proba_calib: np.ndarray, y_calib: np.ndarray, target_coverage: float = 0.9
    ) -> ConformalCalibrationSummary:
        y_idx = np.array([self.class_index[label] for label in y_calib])
        if self.method == "aps":
            scores = _aps_scores(proba_calib, y_idx, self._calib_rng)
        else:
            scores = _lac_scores(proba_calib, y_idx)

        n = len(scores)
        alpha = 1.0 - target_coverage
        # Finite-sample-correct empirical quantile (Vovk et al.): ceil((n+1)(1-alpha)) / n.
        level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        self.q_hat_ = float(np.quantile(scores, level, method="higher"))
        self.target_coverage = target_coverage
        return ConformalCalibrationSummary(self.method, target_coverage, self.q_hat_, n)

    def predict_sets(self, proba_test: np.ndarray) -> np.ndarray:
        if self.q_hat_ is None:
            raise RuntimeError("call calibrate() before predict_sets()")
        if self.method == "aps":
            return _aps_sets(proba_test, self.q_hat_, self._predict_rng)
        return _lac_sets(proba_test, self.q_hat_)

    def predict_set_labels(self, proba_test: np.ndarray) -> list[list[str]]:
        bool_sets = self.predict_sets(proba_test)
        return [[c for c, included in zip(self.classes, row) if included] for row in bool_sets]


def empirical_coverage(bool_sets: np.ndarray, y_true: np.ndarray, classes: list[str]) -> float:
    class_index = _class_index_map(classes)
    y_idx = np.array([class_index[label] for label in y_true])
    covered = bool_sets[np.arange(len(y_idx)), y_idx]
    return float(covered.mean())


def coverage_by_class(bool_sets: np.ndarray, y_true: np.ndarray, classes: list[str]) -> dict[str, float]:
    class_index = _class_index_map(classes)
    result = {}
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            result[c] = float("nan")
            continue
        result[c] = float(bool_sets[mask, class_index[c]].mean())
    return result


def average_set_size(bool_sets: np.ndarray) -> float:
    return float(bool_sets.sum(axis=1).mean())


def abstention_rate(bool_sets: np.ndarray) -> float:
    """Fraction of predictions with an empty set. Common with LAC on a
    low-confidence window; rare but possible with randomized APS too (only
    when even the single most likely class's own coin flip excludes it),
    unlike the deterministic APS variant which always includes at least the
    top-1 class."""
    return float((bool_sets.sum(axis=1) == 0).mean())
