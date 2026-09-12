#!/usr/bin/env python3
"""Generate standalone PNG figures for the slide decks and the
Architecture+Schedule PDF -- reuses the same audit/pipeline/evaluation code
the notebooks call, so these figures never drift from what the pipeline
actually measured.

Usage: python scripts/generate_deliverable_assets.py
Outputs to docs/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pdm.config import REPO_ROOT, load_config
from pdm.data.audit import run_audit
from pdm.data.loader import load_sensor_dataset
from pdm.evaluation.conformal import average_set_size, empirical_coverage
from pdm.evaluation.metrics import confusion_matrix_df, evaluate_calibration
from pdm.evaluation.robustness import sensor_dropout_test
from pdm.models.bundle import BUNDLE_FILENAME, ModelBundle
from pdm.models.pipeline import evaluate_bundle_on_holdout

OUT = REPO_ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["figure.dpi"] = 150
STYLE = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
plt.style.use(STYLE)

PRIMARY = "#4C72B0"
ACCENT = "#C44E52"
GOOD = "#55A868"


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def class_balance_figure(dataset) -> None:
    classes, counts = np.unique(dataset.labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(classes, counts, color=PRIMARY)
    ax.set_ylabel("Janelas")
    ax.set_title("Balanceamento de classes no dataset do case")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 200, str(c), ha="center")
    save(fig, "class_balance.png")


def sensor_verdict_figure(report) -> None:
    names = list(report.sensor_audits)
    retained = [n in report.recommended_sensors for n in names]
    colors = [GOOD if r else ACCENT for r in retained]
    mi = [report.sensor_audits[n].mutual_info_with_label for n in names]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(names, mi, color=colors)
    ax.set_ylabel("Informação mútua com a classe")
    ax.set_title("Auditoria de sensores: retidos (verde) vs. excluídos (vermelho)")
    save(fig, "sensor_audit_verdict.png")


def confusion_matrix_figure(bundle, holdout) -> None:
    cm = confusion_matrix_df(holdout["y_test"], holdout["y_pred"], bundle.classes)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(bundle.classes)))
    ax.set_xticklabels(bundle.classes, rotation=45)
    ax.set_yticks(range(len(bundle.classes)))
    ax.set_yticklabels(bundle.classes)
    for i in range(len(bundle.classes)):
        for j in range(len(bundle.classes)):
            ax.text(j, i, cm.values[i, j], ha="center", va="center",
                     color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title(f"Matriz de confusão -- {bundle.model_name}")
    save(fig, "confusion_matrix.png")


def calibration_figure(bundle, holdout) -> None:
    calibration = evaluate_calibration(holdout["y_test"], holdout["proba"], bundle.classes)
    mask = calibration.bin_count > 0
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="calibração perfeita")
    ax.plot(calibration.bin_confidence[mask], calibration.bin_accuracy[mask], "o-", color=PRIMARY)
    ax.set_xlabel("Confiança prevista")
    ax.set_ylabel("Acurácia observada")
    ax.set_title(f"Calibração (ECE={calibration.expected_calibration_error:.4f})")
    ax.legend()
    save(fig, "calibration.png")


def conformal_figure(bundle, holdout) -> None:
    sets = holdout["conformal_sets"]
    if sets is None:
        return
    coverage = empirical_coverage(sets, holdout["y_test"], bundle.classes)
    avg_size = average_set_size(sets)
    set_sizes = sets.sum(axis=1)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(set_sizes, bins=np.arange(0, len(bundle.classes) + 2) - 0.5, color=GOOD, rwidth=0.8)
    ax.set_xticks(range(len(bundle.classes) + 1))
    ax.set_xlabel("Tamanho do conjunto de predição conformal")
    ax.set_ylabel("Nº de janelas de teste")
    ax.set_title(f"Cobertura={coverage:.3f}  |  Tamanho médio={avg_size:.2f}")
    save(fig, "conformal_set_sizes.png")


def robustness_figure(bundle, holdout) -> None:
    results = sensor_dropout_test(
        bundle.model, holdout["X_test"], holdout["y_test"], bundle.classes, bundle.sensor_names
    )
    labels = [r.scenario for r in results]
    values = [r.f1_macro for r in results]
    colors = [PRIMARY] + [ACCENT] * (len(results) - 1)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("F1-macro")
    ax.set_xlim(0, 1)
    ax.set_title("Degradação ao remover um sensor")
    save(fig, "robustness_sensor_dropout.png")


def controls_figure(bundle) -> None:
    labels = ["F1 real (teste)", "Controle: rótulos\nembaralhados", "Controle: só sensor\nde ruído excluído", "Acaso teórico\n(5 classes)"]
    values = [
        bundle.metrics.get("test_f1_macro", float("nan")),
        bundle.metrics.get("label_shuffle_control_f1", float("nan")),
        bundle.metrics.get("noise_sensor_control_f1", float("nan")),
        1 / len(bundle.classes),
    ]
    colors = [GOOD, "#8C8C8C", "#8C8C8C", ACCENT]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 1)
    ax.set_title("O resultado é real: controles caem para o nível do acaso")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    save(fig, "sanity_controls.png")


def gantt_figure() -> None:
    phases = [
        ("Fase 0 - Requisitos", 0, 1, "#4C72B0"),
        ("Fase 1 - Entendimento", 1, 1, "#4C72B0"),
        ("Fase 2 - Baseline", 2, 2, "#55A868"),
        ("Fase 3 - Pesquisa avançada", 3, 3, "#DD8452"),
        ("Fase 4 - Integração e piloto", 6, 1.5, "#8172B2"),
        ("Fase 5 - Validação de campo", 7.5, 1, "#C44E52"),
        ("Fase 6 - Acompanhamento", 8.5, 3, "#8C8C8C"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, start, dur, color) in enumerate(phases):
        ax.barh(name, dur, left=start, color=color, height=0.6)
    ax.set_xlabel("Mês do projeto")
    ax.set_xlim(0, 12)
    ax.set_title("Cronograma de execução -- 9 meses (Fase 6 inicia sustentação contínua)")
    ax.invert_yaxis()
    save(fig, "gantt_chart.png")


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    boxes = {
        "Banco do cliente\n(automação/software)": (0.05, 0.55, 0.18, 0.22, "#8C8C8C"),
        "Ingestão\n(loader + audit)": (0.30, 0.55, 0.18, 0.22, PRIMARY),
        "Pré-proc. + Features\n(tempo/freq/wavelet)": (0.55, 0.55, 0.20, 0.22, PRIMARY),
        "Modelo + Conformal\n(ModelBundle)": (0.55, 0.20, 0.20, 0.22, GOOD),
        "API (FastAPI)": (0.30, 0.20, 0.18, 0.22, GOOD),
        "Banco de predições\n/auditoria/drift": (0.05, 0.20, 0.18, 0.22, "#8C8C8C"),
        "Dashboard\n(Streamlit)": (0.80, 0.20, 0.16, 0.22, "#DD8452"),
    }
    for label, (x, y, w, h, color) in boxes.items():
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", alpha=0.85))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, color="white", weight="bold")

    arrows = [
        ((0.23, 0.66), (0.30, 0.66)),
        ((0.48, 0.66), (0.55, 0.66)),
        ((0.65, 0.55), (0.65, 0.42)),
        ((0.55, 0.31), (0.48, 0.31)),
        ((0.30, 0.31), (0.23, 0.31)),
        ((0.48, 0.31), (0.80, 0.31)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Arquitetura do sistema (ver docs/03_arquitetura.md para diagramas C4 completos)")
    save(fig, "architecture_overview.png")


def main() -> None:
    cfg = load_config()
    dataset = load_sensor_dataset(cfg)
    report = run_audit(dataset, cfg)
    class_balance_figure(dataset)
    sensor_verdict_figure(report)
    gantt_figure()
    architecture_figure()

    bundle_path = cfg.paths.models_dir / BUNDLE_FILENAME
    if bundle_path.exists():
        bundle = ModelBundle.load(bundle_path)
        holdout = evaluate_bundle_on_holdout(cfg, bundle, dataset)
        confusion_matrix_figure(bundle, holdout)
        calibration_figure(bundle, holdout)
        conformal_figure(bundle, holdout)
        robustness_figure(bundle, holdout)
        controls_figure(bundle)
    else:
        print(f"No bundle at {bundle_path} -- skipping model-dependent figures.")


if __name__ == "__main__":
    main()
