# Predictive Maintenance -- Electric Motor (SENAI SC case study)

A complete, end-to-end predictive-maintenance pipeline for a rotating
electric-motor asset: data audit, preprocessing, feature engineering
(time / frequency / wavelet), model selection, robustness and
explainability evaluation, and a served API + database + dashboard.

**Result on the case dataset** (full run on the Linux x86-64 target, complete
stack -- MLflow, real SHAP, ssqueezepy; see `docs/03_arquitetura.md` section
3.7 for the cross-platform comparison table): HistGradientBoosting (selected
by cross-validation among 4 candidates, on both platforms tested) reaches
**F1-macro = 0.962** on a held-out test split, with expected calibration
error **0.006** and conformal-prediction coverage **0.894** against a 0.90
target (average prediction-set size 0.98). Two sanity controls confirm this
is real signal, not leakage: label-shuffling lands at F1 = 0.200 (exactly
chance) and a model trained on the audit-excluded noise sensor alone scores
F1 = 0.067 -- below chance. See
[`docs/01_interpretacao_problema.md`](docs/01_interpretacao_problema.md)
for the full interpretation and [`notebooks/`](notebooks/) for the executed
analysis behind these numbers.

Resumo em português: pipeline completo de manutenção preditiva para um
motor elétrico -- auditoria de dados, pré-processamento, engenharia de
features, seleção e avaliação de modelos, e disponibilização via API,
banco de dados e dashboard. A documentação de negócio (interpretação do
problema, arquitetura, cronograma, riscos e track de pesquisa) está em
[`docs/`](docs/).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

python scripts/download_data.py  # fetch Classes.npy + Dados_1..5.npy (~380 MB, gitignored)
python -m pdm.cli audit          # data-quality audit
python -m pdm.cli features       # cache the engineered feature table
python -m pdm.cli train          # audit -> features -> model selection -> ModelBundle
python -m pdm.cli evaluate       # print the saved bundle's metrics
python -m pdm.cli serve          # FastAPI inference service on :8000
```

With the API running, `streamlit run src/pdm/serving/dashboard.py` starts the
dashboard, or `docker compose -f docker/docker-compose.yml up --build` runs
the API + dashboard + Postgres stack in containers.

## Notebooks

- [`01_data_audit_and_signal_analysis.ipynb`](notebooks/01_data_audit_and_signal_analysis.ipynb) -- the data audit's findings, visualized (class balance, the ghost column, saturation/Hampel filtering, silent windows, the two excluded sensors, per-class spectra, a wavelet scalogram).
- [`02_feasibility_and_evaluation.ipynb`](notebooks/02_feasibility_and_evaluation.ipynb) -- confusion matrix, calibration, conformal prediction sets, imbalance-aware re-evaluation, robustness (sensor dropout / feature-group ablation), explainability.
- [`03_api_demo.ipynb`](notebooks/03_api_demo.ipynb) -- runs the real FastAPI service, exercises every endpoint (`/predict`, `/predict_batch`, `/explain`, `/audit`, `/drift`), and reads the resulting database directly with pandas.

All three run against the real case dataset and are committed with their
outputs already executed.

## Repository layout

```
src/pdm/          Library code (data, preprocessing, features, models, evaluation, serving)
notebooks/        Executed notebooks -- audit/signal analysis, evaluation, API+DB demo
docs/             Problem interpretation, requirements, architecture, schedule, risks, research track
tests/            pytest suite (runs against a small synthetic fixture, not the real data)
docker/           Container images for the API, dashboard and database
scripts/          Data download, deliverable asset/slide/PDF generation, docx template filler
configs/          YAML configuration (paths, thresholds, seeds)
```

## Platform notes

Developed on Windows ARM64 and continuously cross-checked on a Linux x86-64
box (both are the declared deployment targets -- see
`docs/03_arquitetura.md`, section 3.6, for the full table). Several
dependencies (`mlflow`, `streamlit`, `shap`, `ssqueezepy`, `psycopg2-binary`,
and `pandas.to_parquet`'s `pyarrow` engine) need a C/C++ toolchain to build
on Windows ARM64, where no prebuilt wheels exist yet -- the code treats
them as optional imports with a documented, tested fallback (a local
JSON+joblib experiment tracker, permutation importance / LIME instead of
SHAP, a plain-PyWavelets CWT instead of the synchrosqueezed transform, a
joblib-pickled feature cache instead of Parquet), and all four were run for
real (not just installed) on Linux x86-64 -- see `docs/03_arquitetura.md`
section 3.7 for that full training run's results, logged to a real MLflow
tracker. `scipy` is pinned to `<1.18` (no lower bound) because 1.18+ hits a
DLL-load failure under a Windows Application Control Policy encountered
during development.

A model bundle is only portable across environments that resolve the same
scikit-learn version: Python 3.10 tops out at scikit-learn 1.7.2, while
Windows ARM64 only gets wheels from 1.9.1 onward (which needs Python
>=3.11), and `HistGradientBoostingClassifier`'s internal loss module moved
between those releases -- a genuinely different-Python-version issue, not a
Windows-vs-Linux one. `requirements.txt` documents this; `ModelBundle.load()`
(`src/pdm/models/bundle.py`) turns the resulting pickle failure into an
actionable error instead of a bare traceback. The bundle committed to this
repo is the one this development environment can load; `docs/03_arquitetura.md`
section 3.7 reports both platforms' numbers side by side.

`docker compose up` was run end-to-end on the Linux target: all three
services (api, dashboard, postgres) start healthy, and a real `POST
/predict` call returns a prediction and persists the row in the
containerized Postgres database, confirmed with a direct `SELECT`.

## License

Proprietary -- prepared as a selection-process case study for SENAI SC.
