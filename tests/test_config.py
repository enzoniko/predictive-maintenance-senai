from __future__ import annotations

from pdm.config import DEFAULT_CONFIG_PATH, load_config


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
    assert cfg.api["database_url"] == "sqlite:///./pdm.db"


def test_default_config_path_exists() -> None:
    assert DEFAULT_CONFIG_PATH.exists()
