from __future__ import annotations

from pdm.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config


def test_default_config_loads_and_has_expected_shape() -> None:
    cfg = load_config()
    assert cfg.acquisition["sample_rate_hz"] == 10000
    assert cfg.acquisition["window_length_samples"] == 200
    assert len(cfg.paths.sensor_files) == 5
    assert cfg.paths.labels_path.name == "Classes.npy"


def test_sensor_path_joins_raw_dir_and_filename() -> None:
    cfg = load_config()
    path = cfg.paths.sensor_path("Dados_1.npy")
    assert path.parent == cfg.paths.raw_dir
    assert path.name == "Dados_1.npy"


def test_env_var_overrides_database_url(monkeypatch) -> None:
    monkeypatch.setenv("PDM_DATABASE_URL", "postgresql+psycopg2://x:y@host/db")
    cfg = load_config()
    assert cfg.api["database_url"] == "postgresql+psycopg2://x:y@host/db"


def test_no_env_var_keeps_yaml_default(monkeypatch) -> None:
    monkeypatch.delenv("PDM_DATABASE_URL", raising=False)
    cfg = load_config()
    # The YAML default is the relative "sqlite:///./pdm.db", but
    # _resolve_sqlite_url anchors it to REPO_ROOT -- see the dedicated test
    # below for why that matters.
    assert cfg.api["database_url"].endswith("pdm.db")
    assert cfg.api["database_url"].startswith("sqlite:///")


def test_default_config_path_exists() -> None:
    assert DEFAULT_CONFIG_PATH.exists()


def test_relative_sqlite_url_is_anchored_to_repo_root(monkeypatch) -> None:
    monkeypatch.delenv("PDM_DATABASE_URL", raising=False)
    cfg = load_config()
    url = cfg.api["database_url"]
    assert url.startswith("sqlite:///")
    # Compare as_posix() on both sides: the URL always uses forward slashes
    # (per _resolve_sqlite_url), while str(Path) uses the OS separator.
    assert REPO_ROOT.as_posix() in url
    # Regression check: this must be an absolute path, not left relative to
    # whatever directory happened to invoke the process (a Jupyter kernel's
    # default working directory is the notebook's own folder, not the repo
    # root -- see config.py's _resolve_sqlite_url docstring).
    resolved_path = url[len("sqlite:///"):]
    from pathlib import Path

    assert Path(resolved_path).is_absolute()


def test_in_memory_sqlite_url_is_left_untouched(monkeypatch) -> None:
    monkeypatch.setenv("PDM_DATABASE_URL", "sqlite:///:memory:")
    cfg = load_config()
    assert cfg.api["database_url"] == "sqlite:///:memory:"


def test_postgres_url_is_left_untouched(monkeypatch) -> None:
    url = "postgresql+psycopg2://pdm:pdm@postgres:5432/pdm"
    monkeypatch.setenv("PDM_DATABASE_URL", url)
    cfg = load_config()
    assert cfg.api["database_url"] == url
