"""Command-line entry points for the pipeline.

    python -m pdm.cli download-data   # fetch the raw .npy files
    python -m pdm.cli audit           # data-quality audit -> docs/reports/
    python -m pdm.cli features        # build & cache the feature table
    python -m pdm.cli train           # train + tune + register models
    python -m pdm.cli evaluate        # full evaluation suite (see docs/03_arquitetura.md)
    python -m pdm.cli serve           # run the FastAPI inference service
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from pdm.config import REPO_ROOT, load_config

app = typer.Typer(add_completion=False, help=__doc__)


@app.command("download-data")
def download_data(url: str = typer.Option(None, help="Override the Drive folder URL")) -> None:
    """Fetch the raw .npy files into data/raw/ (wraps scripts/download_data.py)."""
    script = REPO_ROOT / "scripts" / "download_data.py"
    cmd = [sys.executable, str(script)]
    if url:
        cmd += ["--url", url]
    raise SystemExit(subprocess.call(cmd))


@app.command("audit")
def audit(
    config_path: Path = typer.Option(None, "--config", help="Path to a config YAML"),
    out: Path = typer.Option(None, help="Where to write the JSON report"),
) -> None:
    """Run the data-quality audit and print + save a human-readable summary."""
    from pdm.data.audit import run_audit
    from pdm.data.loader import load_sensor_dataset

    cfg = load_config(config_path)
    dataset = load_sensor_dataset(cfg)
    report = run_audit(dataset, cfg)

    out_path = out or (cfg.paths.reports_dir / "audit_report.json")
    report.save(out_path)

    typer.echo(f"Windows: {report.n_windows}  Window length: {report.window_len} samples "
               f"@ {report.sample_rate_hz} Hz")
    typer.echo(f"Class balance: {report.class_balance_verdict}")
    typer.echo(f"Row contiguity: {report.row_contiguity.verdict}")
    typer.echo(f"Simultaneity: {report.inter_sensor_simultaneity.verdict}")
    typer.echo("")
    for name, sa in report.sensor_audits.items():
        typer.echo(f"  {name}: {sa.verdict}")
    typer.echo("")
    typer.echo(f"Recommended sensors: {report.recommended_sensors}")
    typer.echo(f"Full report saved to {out_path}")


@app.command("features")
def features_cmd(
    config_path: Path = typer.Option(None, "--config", help="Path to a config YAML"),
) -> None:
    """Build and cache the engineered feature table (time + frequency + wavelet)."""
    from pdm.data.audit import run_audit
    from pdm.data.io import save_feature_table
    from pdm.data.loader import load_sensor_dataset
    from pdm.features.builder import build_feature_table

    cfg = load_config(config_path)
    dataset = load_sensor_dataset(cfg)
    report = run_audit(dataset, cfg)
    table = build_feature_table(dataset, cfg, sensors=report.recommended_sensors)

    out_path = save_feature_table(table, cfg.paths.processed_dir)
    typer.echo(f"Feature table: {table.shape} -> {out_path}")


@app.command("train")
def train_cmd(
    config_path: Path = typer.Option(None, "--config", help="Path to a config YAML"),
    out: Path = typer.Option(None, help="Where to save the model bundle"),
) -> None:
    """Run the full pipeline (audit -> features -> model selection ->
    conformal calibration) and save a deployable ModelBundle."""
    from pdm.models.bundle import BUNDLE_FILENAME
    from pdm.models.pipeline import run_training_pipeline
    from pdm.models.registry import MLFLOW_AVAILABLE, get_tracker

    cfg = load_config(config_path)
    typer.echo("Running audit + feature engineering + model selection (this can take "
               "several minutes on the full dataset)...")
    artifacts = run_training_pipeline(cfg)

    out_path = out or (cfg.paths.models_dir / BUNDLE_FILENAME)
    artifacts.bundle.save(out_path)

    tracker = get_tracker(cfg.paths.models_dir / "runs")
    run_id = tracker.start_run(f"train-{artifacts.best_model_name}")
    tracker.log_params(run_id, {
        "model_name": artifacts.best_model_name,
        "sensors": ",".join(artifacts.bundle.sensor_names),
        "random_seed": cfg.random_seed,
        "conformal_method": cfg.conformal["method"],
        "conformal_target_coverage": cfg.conformal["target_coverage"],
    })
    numeric_metrics = {k: v for k, v in artifacts.bundle.metrics.items() if isinstance(v, (int, float))}
    tracker.log_metrics(run_id, numeric_metrics)
    try:
        # mlflow.sklearn.log_model expects a real scikit-learn estimator;
        # models/train.py::EncodedLabelClassifier (the XGBoost wrapper) is
        # not one, so this is best-effort -- the ModelBundle joblib file
        # (already saved above) remains the source of truth serving/api.py
        # actually loads, regardless of whether this logging step succeeds.
        tracker.log_model(run_id, artifacts.bundle.model)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see comment above
        typer.echo(f"Warning: could not log the model artifact to the tracker ({exc})")
    typer.echo(f"Logged to {'MLflow' if MLFLOW_AVAILABLE else 'local tracker'} (run {run_id}) "
               f"under {cfg.paths.models_dir / 'runs'}")

    typer.echo(f"Best model: {artifacts.best_model_name}")
    typer.echo(f"Test macro F1: {artifacts.test_classification_report['macro avg']['f1-score']:.4f}")
    typer.echo(f"Test ECE: {artifacts.test_calibration.expected_calibration_error:.4f}")
    typer.echo(f"Conformal coverage @ target {cfg.conformal['target_coverage']}: "
               f"{artifacts.conformal_coverage:.4f} (avg set size {artifacts.conformal_avg_set_size:.2f})")
    typer.echo(f"Controls -- label shuffle F1: {artifacts.label_shuffle_control_f1:.4f}, "
               f"excluded-sensor F1: {artifacts.noise_sensor_control_f1:.4f} (chance ~= "
               f"{1 / len(artifacts.bundle.classes):.4f})")
    typer.echo(f"Bundle saved to {out_path}")


@app.command("evaluate")
def evaluate_cmd(
    config_path: Path = typer.Option(None, "--config", help="Path to a config YAML"),
    bundle_path: Path = typer.Option(None, help="Path to a saved model bundle"),
) -> None:
    """Print the evaluation summary for a saved bundle (re-runs the pipeline
    if no cached artifacts are found -- see also `python -m pdm.cli train`,
    which prints the same summary right after training)."""
    from pdm.models.bundle import BUNDLE_FILENAME, ModelBundle

    cfg = load_config(config_path)
    path = bundle_path or (cfg.paths.models_dir / BUNDLE_FILENAME)
    if not path.exists():
        typer.echo(f"No bundle at {path} -- run `python -m pdm.cli train` first.")
        raise typer.Exit(1)

    bundle = ModelBundle.load(path)
    typer.echo(f"Model: {bundle.model_name}  Trained: {bundle.created_at}")
    for k, v in bundle.metrics.items():
        typer.echo(f"  {k}: {v}")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option(None),
    port: int = typer.Option(None),
) -> None:
    """Run the FastAPI inference service with uvicorn."""
    cfg = load_config()
    import uvicorn

    uvicorn.run(
        "pdm.serving.api:app",
        host=host or cfg.api["host"],
        port=port or cfg.api["port"],
        reload=False,
    )


if __name__ == "__main__":
    app()
