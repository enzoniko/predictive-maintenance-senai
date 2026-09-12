"""Command-line entry points for the pipeline.

    python -m pdm.cli download-data   # fetch the raw .npy files
    python -m pdm.cli audit           # data-quality audit -> docs/reports/
    python -m pdm.cli features        # build & cache the feature table
    python -m pdm.cli train           # train + tune + register models
    python -m pdm.cli evaluate        # full evaluation suite (see docs/03_arquitetura.md)
    python -m pdm.cli serve           # run the FastAPI inference service
"""

from __future__ import annotations

import json
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
    from pdm.data.loader import load_sensor_dataset
    from pdm.features.builder import build_feature_table

    cfg = load_config(config_path)
    dataset = load_sensor_dataset(cfg)
    report = run_audit(dataset, cfg)
    table = build_feature_table(dataset, cfg, sensors=report.recommended_sensors)

    out_path = cfg.paths.processed_dir / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path)
    typer.echo(f"Feature table: {table.shape} -> {out_path}")


@app.command("train")
def train_cmd() -> None:
    """Train, tune and register the candidate models. See src/pdm/models/train.py."""
    typer.echo("Not yet wired into the CLI -- run notebooks/03_feasibility_modeling.ipynb "
               "or `python -m pdm.models.train` directly for now.")


@app.command("evaluate")
def evaluate_cmd() -> None:
    """Run the full evaluation suite. See src/pdm/evaluation/."""
    typer.echo("Not yet wired into the CLI -- see notebooks/04_evaluation_robustness_"
               "explainability.ipynb.")


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
