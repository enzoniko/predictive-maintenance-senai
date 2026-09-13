#!/usr/bin/env python3
"""Generate standalone PNG figures for the slide decks and the
Architecture+Schedule PDF. Reuses the same audit/pipeline/evaluation code
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


def save(fig, name: str, dpi: int | None = None) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
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
    ax.set_title(f"Matriz de confusão: {bundle.model_name}")
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
        ("Fase 0: Requisitos", 0, 1, "#4C72B0"),
        ("Fase 1: Entendimento", 1, 1, "#4C72B0"),
        ("Fase 2: Baseline", 2, 2, "#55A868"),
        ("Fase 3: Integração e piloto", 4, 2, "#8172B2"),
        ("Fase 4: Validação de campo", 6, 1, "#C44E52"),
        ("Fase 5: Acompanhamento", 7, 2, "#8C8C8C"),
        ("Trilha de pesquisa interna\n(equipe de IA do SENAI, paralela)", 2, 3, "#DD8452"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, start, dur, color in phases:
        ax.barh(name, dur, left=start, color=color, height=0.6)
    ax.set_xlabel("Mês do projeto")
    ax.set_xlim(0, 9)
    ax.set_title("Cronograma de execução: 9 meses (Fase 5 inicia sustentação contínua)")
    ax.invert_yaxis()
    save(fig, "gantt_chart.png")


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off")
    boxes = {
        "Banco do cliente\n(automação/software)\nsensores + placas": (0.03, 0.55, 0.18, 0.26, "#8C8C8C"),
        "Ingestão\nloader.py + audit.py\nschema, Kruskal-Wallis, MI": (0.28, 0.55, 0.20, 0.26, PRIMARY),
        "Features\ntempo + frequência + wavelet\nbuilder.py (fit/transform)": (0.55, 0.55, 0.20, 0.26, PRIMARY),
        "Modelo + Conformal\nModelBundle (joblib)": (0.55, 0.16, 0.20, 0.26, GOOD),
        "API (FastAPI)\n/predict /predict_batch\n/explain /audit /drift": (0.28, 0.16, 0.20, 0.26, GOOD),
        "Banco de predições\n/auditoria/drift\nSQLite ou Postgres": (0.03, 0.16, 0.18, 0.26, "#8C8C8C"),
        "Dashboard\nStreamlit\ntime de manutenção": (0.81, 0.16, 0.16, 0.26, "#DD8452"),
    }
    for label, (x, y, w, h, color) in boxes.items():
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", alpha=0.85))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5, color="white", weight="bold")

    arrows = [
        ((0.21, 0.68), (0.28, 0.68)),
        ((0.48, 0.68), (0.55, 0.68)),
        ((0.65, 0.55), (0.65, 0.42)),
        ((0.55, 0.29), (0.48, 0.29)),
        ((0.28, 0.29), (0.21, 0.29)),
        ((0.48, 0.29), (0.81, 0.29)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Arquitetura do sistema (ver docs/03_arquitetura.md para os diagramas C4 completos)")
    save(fig, "architecture_overview.png")


def pipeline_detail_figure() -> None:
    """A second, more granular diagram: the concrete steps inside the
    training pipeline (python -m pdm.cli train), one level below the
    component-level architecture_overview.png above.
    """
    fig, ax = plt.subplots(figsize=(16, 4.2))
    ax.axis("off")
    steps = [
        ("Dados\nbrutos", "#8C8C8C"),
        ("Auditoria\nestatística", PRIMARY),
        ("Split\nestratificado", PRIMARY),
        ("Limpeza\n(fit no treino)", PRIMARY),
        ("Features\n(tempo/freq/\nwavelet)", PRIMARY),
        ("Seleção de\nmodelo (CV)", GOOD),
        ("Calibração\nconformal", GOOD),
        ("ModelBundle", "#DD8452"),
    ]
    captions = [
        ".npy / banco do cliente",
        "Kruskal-Wallis + MI",
        "80/10/10",
        "Hampel, NaN, silêncio",
        "CWT / synchrosqueezed",
        "5-fold, 4 candidatos",
        "LAC / APS",
        "modelo + limpadores + metadados",
    ]
    n = len(steps)
    box_w, box_h, gap = 0.105, 0.42, 0.02
    total_w = n * box_w + (n - 1) * gap
    x0 = (1 - total_w) / 2
    y0 = 0.42
    for i, ((label, color), caption) in enumerate(zip(steps, captions)):
        x = x0 + i * (box_w + gap)
        ax.add_patch(plt.Rectangle((x, y0), box_w, box_h, facecolor=color, edgecolor="black", alpha=0.88))
        ax.text(x + box_w / 2, y0 + box_h / 2, label, ha="center", va="center",
                fontsize=10, color="white", weight="bold")
        ax.text(x + box_w / 2, y0 - 0.05, caption, ha="center", va="top",
                fontsize=7.3, color="#333333", wrap=True)
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap, y0 + box_h / 2), xytext=(x + box_w, y0 + box_h / 2),
                        arrowprops=dict(arrowstyle="->", lw=1.4, color="black"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Pipeline de treino em detalhe (python -m pdm.cli train), ver docs/03_arquitetura.md seção 3.3")
    save(fig, "pipeline_detail.png")


def wavelet_feature_figure(dataset, fs: int, window_len: int) -> None:
    """Visualizes the wavelet feature-extraction step directly: a CWT
    scalogram of one real window per class, on the sensor with the largest
    amplitude among the three retained. Mirrors the analysis in
    notebooks/01_data_audit_and_signal_analysis.ipynb.
    """
    import pywt

    sensor = "Dados_2"
    scales = np.geomspace(1, 128, 60)
    class_names = sorted(set(dataset.labels.tolist()))
    fig, axes = plt.subplots(1, len(class_names), figsize=(16, 3.2), sharey=True)
    for ax, cls in zip(axes, class_names):
        idx = np.where(dataset.labels == cls)[0][0]
        row = dataset.sensors[sensor][idx]
        coeffs, cwt_freqs = pywt.cwt(row, scales, "morl", sampling_period=1 / fs)
        ax.imshow(
            np.abs(coeffs), aspect="auto",
            extent=[0, window_len / fs * 1000, cwt_freqs[-1], cwt_freqs[0]],
            cmap="magma",
        )
        ax.set_title(cls, fontsize=9)
        ax.set_xlabel("ms")
    axes[0].set_ylabel("Hz (aprox.)")
    fig.suptitle(f"Escalograma CWT (Morlet): {sensor}, uma janela de exemplo por classe", y=1.05)
    save(fig, "wavelet_scalogram.png", dpi=90)


def main() -> None:
    cfg = load_config()
    dataset = load_sensor_dataset(cfg)
    report = run_audit(dataset, cfg)
    class_balance_figure(dataset)
    sensor_verdict_figure(report)
    gantt_figure()
    architecture_figure()
    pipeline_detail_figure()
    wavelet_feature_figure(dataset, dataset.sample_rate_hz, dataset.window_len)

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
        print(f"No bundle at {bundle_path}, skipping model-dependent figures.")


if __name__ == "__main__":
    main()
