"""Streamlit dashboard for the maintenance team: predict on a window, see
the model's uncertainty and explanation, watch the recent prediction
history and the data-drift monitor, and re-run the data-quality audit.

Talks to the FastAPI service over HTTP (not to the model directly), so it
exercises the exact same code path a real client would, and can run on a
different machine from the API. ``streamlit`` needs a C/C++ toolchain to
build on some environments (see docs/03_arquitetura.md, section 3.6);
this file was validated on the Linux test server / inside Docker.

Run with:  streamlit run src/pdm/serving/dashboard.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("PDM_API_URL", "http://localhost:8000")
SAMPLE_WINDOWS_PATH = Path(__file__).resolve().parents[3] / "configs" / "sample_windows.json"

st.set_page_config(page_title="Predictive Maintenance", layout="wide", page_icon="\U0001F527")
st.title("Manutenção Preditiva: Motor Elétrico")
st.caption(
    "Dashboard operacional do time de manutenção. Fala com a API real por HTTP; "
    "nenhuma predição aqui é calculada localmente."
)


@st.cache_data(ttl=30)
def get_health() -> dict:
    resp = requests.get(f"{API_URL}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data
def load_real_examples() -> dict | None:
    """One real window per class from the case dataset, bundled as a small
    committed asset (see scripts that regenerate configs/sample_windows.json)
    so the dashboard has genuine sensor data to demo with even where the
    full ~380 MB raw dataset is not mounted (e.g. the Docker image)."""
    if not SAMPLE_WINDOWS_PATH.exists():
        return None
    with open(SAMPLE_WINDOWS_PATH, encoding="utf-8") as f:
        return json.load(f)


def api_error_detail(exc: requests.RequestException) -> str:
    """FastAPI's HTTPException puts the actionable message in the response
    body's ``detail`` field; ``requests``' own exception text is just the
    HTTP status line ("503 Server Error: ..."), which throws away exactly
    the part meant to tell the operator what to do next."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        if detail:
            return str(detail)
    return str(exc)


def synthetic_window(window_len: int, seed: int) -> np.ndarray:
    """Pure noise, offered only as an explicit stress test (a client sending
    garbage), never as the default demo input; see the real-example and
    CSV-upload sources for genuine sensor data."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.15, size=window_len)


try:
    health = get_health()
except requests.RequestException as exc:
    st.error(f"Could not reach the API at {API_URL}: {exc}\n\nStart it with "
             "`python -m pdm.cli serve` or `docker compose up api`.")
    st.stop()

st.sidebar.header("Modelo")
st.sidebar.write(f"**Modelo:** {health['model_name']}")
st.sidebar.write(f"**Classes:** {', '.join(health['classes'])}")
st.sidebar.write(f"**Sensores:** {', '.join(health['sensor_names'])}")
st.sidebar.write(f"**Janela:** {health['window_len']} amostras @ {health['sample_rate_hz']} Hz")

st.sidebar.header("Janela de entrada")
real_examples = load_real_examples()
source_options = ["Upload CSV", "Ruído sintético (teste de estresse)"]
if real_examples:
    source_options.insert(0, "Exemplo real (uma janela por classe)")
source = st.sidebar.radio("Origem dos dados", source_options)

sensors: dict[str, list[float]] = {}
gain_fault = False
if source == "Exemplo real (uma janela por classe)":
    chosen_class = st.sidebar.selectbox("Classe de exemplo", list(real_examples["examples"]))
    sensors = {name: list(vals) for name, vals in real_examples["examples"][chosen_class].items()}
    st.sidebar.caption(
        "Janela real do dataset do case, retida pela auditoria estatística "
        "(ver docs/01_interpretacao_problema.md)."
    )
    gain_fault = st.sidebar.checkbox(
        "Simular defeito de sensor (ganho +80% em Dados_1)",
        help="Reproduz a manipulação usada em notebooks/03_api_demo.ipynb para provocar "
        "um alarme de drift real; útil para ver o sistema reagir a um desvio genuíno.",
    )
    if gain_fault and "Dados_1" in sensors:
        sensors["Dados_1"] = (np.array(sensors["Dados_1"]) * 1.8).tolist()
elif source == "Upload CSV":
    st.sidebar.caption(
        "CSV com uma coluna por sensor "
        f"({', '.join(health['sensor_names'])}), {health['window_len']} linhas."
    )
    uploaded = st.sidebar.file_uploader("Enviar CSV da janela", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in health["sensor_names"] if c not in df.columns]
        if missing:
            st.sidebar.error(f"CSV sem as colunas: {missing}")
        else:
            for name in health["sensor_names"]:
                sensors[name] = df[name].tolist()[: health["window_len"]]
else:
    seed = st.sidebar.number_input("Seed aleatória", value=0, step=1)
    for name in health["sensor_names"]:
        sensors[name] = synthetic_window(health["window_len"], seed + hash(name) % 1000).tolist()

tab_predict, tab_history, tab_drift, tab_audit = st.tabs(
    ["Predição", "Histórico recente", "Monitoramento de drift", "Auditoria de dados"]
)

with tab_predict:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Sinal")
        if sensors:
            chart_df = pd.DataFrame(sensors)
            st.line_chart(chart_df)
            if gain_fault:
                st.caption("Dados_1 amplificado artificialmente (defeito simulado).")
        else:
            st.info("Escolha uma origem de dados na barra lateral para ver o sinal aqui.")

    with col2:
        st.subheader("Predição")
        if sensors and st.button("Prever", type="primary"):
            try:
                resp = requests.post(f"{API_URL}/predict", json={"sensors": sensors}, timeout=30)
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as exc:
                st.error(f"Falha na predição: {api_error_detail(exc)}")
            else:
                st.metric("Classe prevista", result["predicted_class"])
                proba_df = pd.DataFrame(
                    {
                        "classe": list(result["probabilities"]),
                        "probabilidade": list(result["probabilities"].values()),
                    }
                ).sort_values("probabilidade", ascending=False)
                st.bar_chart(proba_df.set_index("classe"))

                st.write(
                    "**Conjunto de predição conformal** (classes que o modelo não consegue "
                    f"descartar estatisticamente no nível de cobertura configurado): "
                    f"{result['conformal_set']}"
                )
                for w in result["warnings"]:
                    st.warning(w)

                try:
                    explain_resp = requests.post(f"{API_URL}/explain", json={"sensors": sensors}, timeout=30)
                    explain_resp.raise_for_status()
                    explanation = explain_resp.json()
                    st.write(f"**Principais variáveis** (método: {explanation['method']}):")
                    st.dataframe(pd.DataFrame(explanation["top_features"]))
                except requests.RequestException as exc:
                    st.caption(f"Explicação indisponível: {api_error_detail(exc)}")

with tab_history:
    st.subheader("Últimas predições registradas no banco de dados")
    n_history = st.slider("Quantidade", min_value=5, max_value=100, value=20, step=5)
    if st.button("Atualizar histórico"):
        st.cache_data.clear()
    try:
        hist_resp = requests.get(f"{API_URL}/predictions/recent", params={"n": n_history}, timeout=30)
        hist_resp.raise_for_status()
        history = hist_resp.json()["predictions"]
    except requests.RequestException as exc:
        st.error(f"Não foi possível carregar o histórico: {api_error_detail(exc)}")
    else:
        if not history:
            st.info("Nenhuma predição registrada ainda nesta sessão da API.")
        else:
            hist_df = pd.DataFrame(history)
            st.dataframe(hist_df, use_container_width=True)
            counts = hist_df["predicted_class"].value_counts()
            st.bar_chart(counts)

with tab_drift:
    st.subheader("Drift de dados: produção recente vs. referência de treino")
    st.caption(
        "Compara as features das predições mais recentes contra a distribuição de treino "
        "(PSI + Kolmogorov-Smirnov, ver docs/09_mlops.md). Exige um número mínimo de "
        "predições recentes para não confundir ruído amostral com drift real."
    )
    if st.button("Verificar drift agora"):
        try:
            drift_resp = requests.get(f"{API_URL}/drift", timeout=60)
            drift_resp.raise_for_status()
            drift = drift_resp.json()
        except requests.RequestException as exc:
            if getattr(exc, "response", None) is not None and exc.response.status_code == 503:
                st.warning(api_error_detail(exc))
            else:
                st.error(f"Falha ao consultar drift: {api_error_detail(exc)}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Alarmes", drift["n_alarms"])
            c2.metric("Janelas de produção comparadas", drift["n_current"])
            c3.metric("Tamanho da referência", drift["n_reference"])
            st.write("**Top features por desvio (PSI):**")
            st.dataframe(pd.DataFrame(drift["top_features"]), use_container_width=True)

with tab_audit:
    st.subheader("Auditoria de qualidade dos dados")
    st.caption("Reexecuta a mesma auditoria estatística do notebook 01, agora via API.")
    if st.button("Rodar auditoria"):
        try:
            audit_resp = requests.get(f"{API_URL}/audit", timeout=120)
            audit_resp.raise_for_status()
            audit = audit_resp.json()
        except requests.RequestException as exc:
            st.error(f"Falha na auditoria: {api_error_detail(exc)}")
        else:
            st.write(f"Janelas analisadas: {audit['n_windows']}")
            st.write(f"Balanceamento de classes: {audit['class_balance_verdict']}")
            st.write(f"Sensores recomendados: {audit['recommended_sensors']}")
            st.write("Sensores excluídos:")
            st.json(audit["excluded_sensors"])
