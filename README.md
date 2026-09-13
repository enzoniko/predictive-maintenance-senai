# Predictive Maintenance for an Electric Motor (SENAI SC case study)

A complete, end-to-end predictive-maintenance pipeline for a rotating
electric-motor asset: data audit, preprocessing, feature engineering
(time / frequency / wavelet), model selection, robustness and
explainability evaluation, and a served API + database + dashboard.

**Result on the case dataset** (full pipeline run, complete stack: MLflow,
real SHAP, ssqueezepy; see `docs/03_arquitetura.md` section 3.7 for the full
run): HistGradientBoosting, selected by cross-validation among 4 candidates,
reaches **F1-macro = 0.962** on a held-out test split, with expected calibration
error **0.006** and conformal-prediction coverage **0.894** against a 0.90
target (average prediction-set size 0.98). Two sanity controls confirm this
is real signal, not leakage: label-shuffling lands at F1 = 0.200, exactly
chance, and a model trained on the audit-excluded noise sensor alone scores
F1 = 0.067, below chance. See
[`docs/01_interpretacao_problema.md`](docs/01_interpretacao_problema.md)
for the full interpretation and [`notebooks/`](notebooks/) for the executed
analysis behind these numbers.

Resumo em português: pipeline completo de manutenção preditiva para um
motor elétrico, com auditoria de dados, pré-processamento, engenharia de
features, seleção e avaliação de modelos, e disponibilização via API,
banco de dados e dashboard. A documentação de negócio (interpretação do
problema, arquitetura, cronograma, riscos e track de pesquisa) está em
[`docs/`](docs/).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/download_data.py  # fetch Classes.npy + Dados_1..5.npy (~380 MB, gitignored)
python -m pdm.cli audit          # data-quality audit
python -m pdm.cli features       # cache the engineered feature table
python -m pdm.cli train          # audit, features, model selection, ModelBundle
python -m pdm.cli evaluate       # print the saved bundle's metrics
python -m pdm.cli serve          # FastAPI inference service on :8000
```

With the API running, `streamlit run src/pdm/serving/dashboard.py` starts the
dashboard, or `docker compose -f docker/docker-compose.yml up --build` runs
the API + dashboard + Postgres stack in containers.

## Notebooks

- [`01_data_audit_and_signal_analysis.ipynb`](notebooks/01_data_audit_and_signal_analysis.ipynb): the data audit's findings, visualized (class balance, the ghost column, saturation/Hampel filtering, silent windows, the two excluded sensors, per-class spectra, a wavelet scalogram).
- [`02_feasibility_and_evaluation.ipynb`](notebooks/02_feasibility_and_evaluation.ipynb): confusion matrix, calibration, conformal prediction sets, imbalance-aware re-evaluation, robustness (sensor dropout / feature-group ablation), explainability.
- [`03_api_demo.ipynb`](notebooks/03_api_demo.ipynb): runs the real FastAPI service, exercises every endpoint (`/predict`, `/predict_batch`, `/explain`, `/audit`, `/drift`), and reads the resulting database directly with pandas.

All three run against the real case dataset and are committed with their
outputs already executed.

## Repository layout

```
src/pdm/          Library code (data, preprocessing, features, models, evaluation, serving)
notebooks/        Executed notebooks: audit/signal analysis, evaluation, API+DB demo
docs/             Problem interpretation, requirements, architecture, schedule, risks, research track
tests/            pytest suite (runs against a small synthetic fixture, not the real data)
docker/           Container images for the API, dashboard and database
scripts/          Data download, decision-tree interpretability analysis
configs/          YAML configuration (paths, thresholds, seeds)
deliverables/     Main case document and executive/technical slide decks
```

## Final deliverables

- [`Enzo Nicolás Spotorno Bieger.docx`](deliverables/Enzo%20Nicolás%20Spotorno%20Bieger.docx): main response, including the architecture, schedule, native WBS and risk tables, and the technical-preview results.
- [`Apresentacao_Executiva.pptx`](deliverables/Apresentacao_Executiva.pptx): 10-slide version for the hiring panel and business stakeholder.
- [`Apresentacao_Tecnica.pptx`](deliverables/Apresentacao_Tecnica.pptx): 22-slide technical defense of the data audit, modeling, uncertainty, robustness and architecture.

## Platform notes

Targets Linux x86-64 in production (see `docs/03_arquitetura.md`, section
3.6). A few dependencies (`mlflow`, `shap`, `ssqueezepy`, `psycopg2-binary`,
and `pandas.to_parquet`'s `pyarrow` engine) rely on compiled extensions that
not every deployment image will have available, so the code treats them as
optional imports with a documented, tested fallback (a local JSON+joblib
experiment tracker, permutation importance / LIME instead of SHAP, a
plain-PyWavelets CWT instead of the synchrosqueezed transform, a
joblib-pickled feature cache instead of Parquet). A missing optional
dependency degrades gracefully instead of crashing the pipeline. The full
stack, all four of the above, for real and not just installed, was run
end-to-end; see `docs/03_arquitetura.md` section 3.7 for that training
run's results, logged to a real MLflow tracker.

A model bundle is only portable across environments that resolve the same
scikit-learn version, since `HistGradientBoostingClassifier`'s internal
loss module has moved between releases. `models/bundle.py::ModelBundle`
records the exact training-environment versions inside the bundle itself,
and `ModelBundle.load()` turns a version mismatch into an actionable error
instead of a bare traceback. Retrain locally with `python -m pdm.cli
train` rather than copying a bundle from a different environment.

`docker compose up` was run end-to-end: all three services (api, dashboard,
postgres) start healthy, and a real `POST /predict` call returns a
prediction and persists the row in the containerized Postgres database,
confirmed with a direct `SELECT`.

## License

Proprietary, prepared as a selection-process case study for SENAI SC.
