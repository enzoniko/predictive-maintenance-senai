from __future__ import annotations

import numpy as np
import pytest

from pdm.config import Config
from pdm.data.loader import SchemaError, load_sensor_dataset


def test_load_sensor_dataset_drops_ghost_columns(synthetic_config: Config) -> None:
    dataset = load_sensor_dataset(synthetic_config)

    assert set(dataset.sensor_names) == {"Dados_1", "Dados_2", "Dados_3", "Dados_4", "Dados_5"}
    for name, matrix in dataset.sensors.items():
        assert matrix.shape == (600, 200), name
    assert set(dataset.ghost_columns) == {"Dados_1", "Dados_2", "Dados_3"}
    assert dataset.n_windows == 600
    assert dataset.class_names() == ["Classe A", "Classe B", "Classe C", "Classe D", "Classe E"]


def test_ghost_column_with_too_much_data_raises(synthetic_config: Config) -> None:
    path = synthetic_config.paths.sensor_path("Dados_1.npy")
    matrix = np.load(path)
    # Corrupt the ghost column so it is no longer near-empty.
    matrix[:, -1] = 1.0
    np.save(path, matrix)

    with pytest.raises(SchemaError, match="refusing to drop it silently"):
        load_sensor_dataset(synthetic_config)


def test_missing_file_raises_clear_error(synthetic_config: Config) -> None:
    synthetic_config.paths.sensor_path("Dados_1.npy").unlink()
    with pytest.raises(FileNotFoundError, match="download-data"):
        load_sensor_dataset(synthetic_config)
