from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from pdm.models.bundle import ModelBundle, _current_environment


def test_current_environment_reports_the_running_interpreter() -> None:
    env = _current_environment()
    assert set(env) == {"python_version", "sklearn_version", "xgboost_version", "platform"}
    assert all(isinstance(v, str) and v for v in env.values())


class _PicklableForTest:
    """A real, importable class; pickled successfully, then its module
    reference is corrupted in the byte stream below so *unpickling* fails
    with ModuleNotFoundError, without needing two scikit-learn versions
    installed side by side to reproduce that failure mode."""


def test_load_wraps_cross_version_unpickle_failures_with_a_clear_message(tmp_path: Path) -> None:
    """Regression test: a bundle trained under one scikit-learn release
    failed with a bare 'ModuleNotFoundError: No module named _loss' when
    loaded under another; HistGradientBoosting's internal loss module
    moved between releases. ModelBundle.load() must turn that into an
    actionable message naming this environment's own versions, rather than
    passing the raw pickle error through.
    """
    real_module = _PicklableForTest.__module__.encode()
    # Protocol 0 is plain-ASCII and newline-delimited, so the embedded
    # module name can be replaced with an arbitrary (nonexistent) one
    # without needing to match its byte length.
    data = pickle.dumps(_PicklableForTest(), protocol=0)
    assert real_module in data  # sanity: the real module name is really embedded
    corrupted = data.replace(real_module, b"definitely_does_not_exist_module_xyz")

    bad_path = tmp_path / "bad_bundle.joblib"
    bad_path.write_bytes(corrupted)

    with pytest.raises(RuntimeError) as exc_info:
        ModelBundle.load(bad_path)
    message = str(exc_info.value)
    assert "Could not load the model bundle" in message
    assert "scikit-learn" in message
    assert "pdm.cli train" in message
