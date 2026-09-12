# Predictive Maintenance -- Electric Motor (SENAI SC case study)

A complete, end-to-end predictive-maintenance pipeline for a rotating
electric-motor asset: data audit, preprocessing, feature engineering
(time / frequency / wavelet), model selection, robustness and
explainability evaluation, and a served API + dashboard backed by a
database.

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

python scripts/download_data.py  # fetch the 6 .npy files (~380 MB, gitignored)
python -m pdm.cli audit          # data-quality audit -> docs/reports/audit_report.json
```

## Repository layout

```
src/pdm/          Library code (data, preprocessing, features, models, evaluation, serving)
notebooks/        Exploratory / feasibility notebooks with executed outputs
docs/             Problem interpretation, requirements, architecture, schedule, risks, research track
tests/            pytest suite (runs against a small synthetic fixture, not the real data)
docker/           Container images for the API, dashboard and database
scripts/          One-off utility scripts (data download, deliverable packaging)
configs/          YAML configuration (paths, thresholds, seeds)
```

## Platform notes

Developed on Windows ARM64 and continuously cross-checked on a Linux x86-64
box; both are supported targets (see `docs/03_arquitetura.md`). A few
dependencies (`mlflow`, `streamlit`, `shap`, `ssqueezepy`) need a C/C++
toolchain to build on Windows ARM64, where no prebuilt wheels exist yet --
the code treats them as optional imports with a documented fallback, and
they are verified working on Linux x86-64. `scipy` is pinned to `<1.18`
because 1.18+ hits a DLL-load failure under a Windows Application Control
Policy encountered during development; the pin has no lower bound so pip
picks the right build per Python version.

## License

Proprietary -- prepared as a selection-process case study for SENAI SC.
