# 3. Arquitetura

## 3.1 Visão de contexto

```mermaid
C4Context
    Person(operador, "Operador / Time de Manutenção", "Consulta predições e alarmes")
    Person(diretor, "Diretor / Stakeholder", "Acompanha valor e risco do projeto")
    System(pdm, "Sistema de Manutenção Preditiva", "Audita dados, treina modelos, serve predições com incerteza calibrada")
    System_Ext(automacao, "Banco de dados do time de Automação/Software", "Sensores e placas -> banco já modelado pelo cliente")
    System_Ext(dashboard_ext, "Dashboards da planta", "Consumo dos alarmes/predições")

    Rel(automacao, pdm, "Fornece janelas de sinais (leitura)")
    Rel(pdm, operador, "Predição + incerteza + explicação")
    Rel(pdm, diretor, "Relatórios de auditoria, drift e valor")
    Rel(pdm, dashboard_ext, "Alimenta via API/banco de predições")
```

## 3.2 Visão de containers (o que existe hoje na prévia)

```mermaid
C4Container
    System_Boundary(pdm, "Sistema de Manutenção Preditiva") {
        Container(cli, "CLI (typer)", "Python", "audit | features | train | evaluate | serve")
        Container(pipeline, "Pipeline de treino", "scikit-learn/XGBoost/Optuna", "Auditoria -> features -> seleção de modelo -> calibração conformal")
        Container(bundle, "ModelBundle", "joblib", "Modelo + limpadores + conformal + metadados, artefato único versionável")
        Container(api, "API de inferência", "FastAPI", "/predict /predict_batch /explain /audit /drift /health")
        ContainerDb(db, "Banco de predições/auditoria/drift", "SQLite (local) / Postgres (docker-compose)", "Só o que a IA produz -- ver 02_engenharia_requisitos.md 2.4")
        Container(dashboard, "Dashboard", "Streamlit", "Visualização para o time de manutenção")
    }
    System_Ext(automacao, "Banco de dados do cliente", "Sensores brutos")

    Rel(cli, pipeline, "invoca")
    Rel(pipeline, automacao, "lê janelas (hoje: scripts/download_data.py simula essa leitura)")
    Rel(pipeline, bundle, "gera")
    Rel(api, bundle, "carrega no startup")
    Rel(api, db, "persiste cada chamada")
    Rel(dashboard, api, "HTTP")
```

## 3.3 Pipeline de treino (o que `python -m pdm.cli train` executa)

```mermaid
flowchart TD
    A[data/loader.py: carrega Classes.npy + Dados_1..5.npy] --> B[data/audit.py: auditoria estatística]
    B -->|sensores recomendados| C[split estratificado 80/10/10 em INDICES brutos]
    C --> D1[fit_cleaners no split de TREINO apenas]
    D1 --> E1[features/builder.py: tempo + frequencia + wavelet, treino]
    D1 --> E2[... calibracao]
    D1 --> E3[... teste]
    E1 --> F[models/train.py: 4 candidatos, CV estratificada 5-fold]
    F --> G[melhor modelo por F1-macro medio]
    G --> H[fit final no treino completo]
    H --> I[evaluation/conformal.py: calibra em E2]
    H --> J[avaliacao completa em E3: metricas, calibracao, imbalance, robustez, controles]
    I --> K[models/bundle.py: ModelBundle]
    J --> K
    K --> L[models/artifacts/model_bundle.joblib]
```

O ponto que mais separa este pipeline de uma implementação ingênua é o passo **D1**: os limiares de limpeza (saturação, silêncio — `preprocessing/cleaning.py`) são ajustados *só* com as linhas de treino e reaplicados sem reajuste em calibração/teste. Nas primeiras versões deste pipeline, esses limiares eram ajustados sobre o conjunto inteiro antes do split — um vazamento sutil (a etapa de limpeza "via" dados de calibração/teste antes de o modelo ser avaliado neles) que o próprio princípio de ceticismo deste projeto (ver `01_interpretacao_problema.md`) exigia corrigir. Ver `src/pdm/features/builder.py` para a separação explícita entre `fit_cleaners` (só treino) e `build_features_from_raw` (aplica limpadores já ajustados).

## 3.4 Pipeline de inferência (o que a API executa por requisição)

```mermaid
sequenceDiagram
    participant C as Cliente (dashboard/planta)
    participant A as API (FastAPI)
    participant F as features/builder.py
    participant M as Modelo (ModelBundle)
    participant Cf as Conformal
    participant DB as Banco

    C->>A: POST /predict {sensores: {Dados_1: [...], ...}}
    A->>A: valida sensores esperados e tamanho de janela
    A->>F: build_features_from_raw(janela, limpadores do bundle)
    F-->>A: vetor de features + flag is_silent
    A->>M: predict_proba(features)
    M-->>A: probabilidades por classe
    A->>Cf: predict_sets(probabilidades)
    Cf-->>A: conjunto de classes plausiveis
    A->>DB: persiste predicao
    A-->>C: classe prevista + probabilidades + conjunto conformal + avisos
```

## 3.5 Mapa do repositório

```
src/pdm/
  config.py          Carrega configs/default.yaml (paths, seeds, limiares)
  cli.py             download-data | audit | features | train | evaluate | serve
  data/              loader.py (schema + validacao), audit.py (auditoria estatistica), io.py (cache com fallback)
  preprocessing/      cleaning.py (Hampel, NaN, silencio), dsp.py (deteccao/filtro/envelope/conversao de unidade)
  features/          time_domain.py, frequency_domain.py, wavelet.py, builder.py (fit/transform separados)
  models/            train.py (candidatos + wrapper XGBoost), tune.py (Optuna), pipeline.py (orquestracao), bundle.py, registry.py (MLflow ou fallback local)
  evaluation/        metrics.py, imbalance.py, robustness.py, explain.py, conformal.py, drift.py
  serving/           api.py, db.py, schemas.py, dashboard.py
notebooks/           01 auditoria, 02 analise de sinal, 03 viabilidade/modelagem, 04 avaliacao/robustez/explicabilidade, 05 demo da API
docs/                este diretorio
tests/               espelha src/pdm/, roda contra um fixture sintetico (nunca os ~380 MB reais)
docker/              Dockerfile.api, Dockerfile.dashboard, docker-compose.yml (api + dashboard + postgres)
```

## 3.6 Notas de plataforma (Windows ARM64 vs. Linux x86-64)

Este projeto foi desenvolvido numa máquina Windows ARM64 e validado continuamente num servidor Linux x86-64 (`ssh niko@niko-g3-3579`, Ubuntu, Docker Engine, GPU disponível para fases futuras de pesquisa) — os dois alvos de implantação declarados nos requisitos de software. Isso expôs, na prática, exatamente o tipo de risco de portabilidade que um projeto industrial real também enfrentaria (ambiente de desenvolvimento do cientista de dados ≠ ambiente de produção do cliente), e cada caso encontrado foi resolvido com um *fallback* documentado no código-fonte, não escondido:

| Dependência | Sintoma no Windows ARM64 | Causa raiz | Solução |
|---|---|---|---|
| `scipy>=1.18` | `DLL load failed` ao importar `scipy.stats` | Política de controle de aplicativos do Windows bloqueia uma extensão compilada específica (`_levy_stable`) | `requirements.txt` fixa `scipy<1.18` sem limite inferior — o pip resolve a versão mais nova compatível por interpretador (1.15.x em Python 3.10, 1.17.x em Python 3.12) |
| `mlflow`, `streamlit`, `shap`, `ssqueezepy` | Falha ao compilar (`pyarrow`/`numba`/`llvmlite` exigem toolchain C/C++ ausente) | Sem *wheel* pré-compilado para Windows ARM64 | Import opcional com *fallback*: `models/registry.py` usa um tracker local (JSON + joblib) quando `mlflow` falta; `evaluation/explain.py` usa importância por permutação quando `shap` falta; `features/wavelet.py` usa PyWavelets puro quando `ssqueezepy` falta. Todos os quatro foram confirmados funcionando no servidor Linux x86-64. |
| `pandas.to_parquet` | `ImportError: Unable to find a usable engine` | Também depende de `pyarrow` | `data/io.py` tenta Parquet primeiro e cai para um DataFrame serializado via `joblib` — mesma API, mesmos pontos de chamada |
| `psycopg2-binary` | Falha ao compilar (exige `pg_config`) | Driver de Postgres, sem *wheel* ARM64 | Não incluído em `requirements.txt` (quebraria a instalação local para uma funcionalidade — Postgres via Docker — que a máquina de desenvolvimento nunca usa); instalado diretamente em `docker/Dockerfile.api`, que só roda em Linux |

Esse padrão — nunca deixar uma dependência ausente quebrar o pipeline inteiro, sempre com uma explicação de causa raiz no código — é o mesmo princípio de ceticismo/robustez aplicado à engenharia de software em vez de aos dados.

## 3.7 Por que não uma rede neural profunda na prévia

Com ~50 mil janelas e um conjunto de features de engenharia de ~150 colunas por sensor retido, este não é um regime de dados escassos que justifique representation learning end-to-end como primeira escolha. Um ensemble de árvores (Random Forest / HistGradientBoosting / XGBoost) já atinge desempenho muito acima do acaso nesses dados (ver `models/artifacts/model_bundle.joblib` e os notebooks 03–04), com a vantagem de manter as features fisicamente nomeadas e a decisão auditável (importância de feature, SHAP, regras de árvore de decisão como baseline interpretável). Deep learning entra no roteiro de pesquisa (`06_track_pesquisa.md`) apenas onde ele tem uma vantagem real e documentada na literatura — aprendizado auto-supervisionado em dados saudáveis abundantes e reconhecimento *open-set* de falhas nunca vistas — não como substituição do que já funciona.
