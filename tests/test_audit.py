from __future__ import annotations

from pdm.config import Config
from pdm.data.audit import run_audit
from pdm.data.loader import load_sensor_dataset


def test_audit_flags_stuck_and_noise_sensors(synthetic_config: Config) -> None:
    dataset = load_sensor_dataset(synthetic_config)
    report = run_audit(dataset, synthetic_config)

    assert report.sensor_audits["Dados_4"].is_stuck
    assert "no information" in report.sensor_audits["Dados_4"].verdict

    assert report.sensor_audits["Dados_5"].is_white_noise
    assert "Dados_5" in report.excluded_sensors

    for name in ("Dados_1", "Dados_2", "Dados_3"):
        assert name in report.recommended_sensors, report.sensor_audits[name]

    assert "perfectly balanced" in report.class_balance_verdict
    assert report.n_windows == 600


def test_ghost_column_report_flags_single_value(synthetic_config: Config) -> None:
    dataset = load_sensor_dataset(synthetic_config)
    report = run_audit(dataset, synthetic_config)

    for name in ("Dados_1", "Dados_2", "Dados_3"):
        check = report.ghost_columns[name]
        assert check.details["n_non_nan"] == 1
        assert "dropped" in check.verdict


def test_report_round_trips_through_json(tmp_path, synthetic_config: Config) -> None:
    dataset = load_sensor_dataset(synthetic_config)
    report = run_audit(dataset, synthetic_config)

    out = tmp_path / "audit_report.json"
    report.save(out)
    assert out.exists()

    import json

    with out.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["n_windows"] == 600
    assert set(payload["sensor_audits"]) == {f"Dados_{i}" for i in range(1, 6)}
