"""Streamlit dashboard: pick or upload a window, see the prediction,
uncertainty and recent history.

Talks to the FastAPI service over HTTP (not to the model directly), so it
exercises the exact same code path a real client would -- and can run on a
different machine from the API. ``streamlit`` needs a C/C++ toolchain to
build on Windows ARM64 (no prebuilt wheel there; see
docs/03_arquitetura.md, "Platform notes") -- this file was validated on the
Linux x86-64 test server / inside Docker, not on the Windows ARM64 dev
machine.

Run with:  streamlit run src/pdm/serving/dashboard.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("PDM_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance -- Electric Motor")


@st.cache_data(ttl=30)
def get_health() -> dict:
    resp = requests.get(f"{API_URL}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def synthetic_window(window_len: int, seed: int) -> np.ndarray:
    """A stand-in for real sensor data, purely so the dashboard has
    something to send when the user has no file handy -- see the sidebar's
    file-upload option for real windows."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.15, size=window_len)


try:
    health = get_health()
except requests.RequestException as exc:
    st.error(f"Could not reach the API at {API_URL}: {exc}\n\nStart it with "
             "`python -m pdm.cli serve` or `docker compose up api`.")
    st.stop()

st.sidebar.header("Model")
st.sidebar.write(f"**Model:** {health['model_name']}")
st.sidebar.write(f"**Classes:** {', '.join(health['classes'])}")
st.sidebar.write(f"**Sensors:** {', '.join(health['sensor_names'])}")
st.sidebar.write(f"**Window:** {health['window_len']} samples @ {health['sample_rate_hz']} Hz")

st.sidebar.header("Input window")
source = st.sidebar.radio("Source", ["Synthetic (demo)", "Upload CSV"])

sensors: dict[str, list[float]] = {}
if source == "Synthetic (demo)":
    seed = st.sidebar.number_input("Random seed", value=0, step=1)
    for name in health["sensor_names"]:
        sensors[name] = synthetic_window(health["window_len"], seed + hash(name) % 1000).tolist()
else:
    st.sidebar.caption(
        "CSV with one column per sensor "
        f"({', '.join(health['sensor_names'])}), {health['window_len']} rows."
    )
    uploaded = st.sidebar.file_uploader("Upload window CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in health["sensor_names"] if c not in df.columns]
        if missing:
            st.sidebar.error(f"CSV is missing columns: {missing}")
        else:
            for name in health["sensor_names"]:
                sensors[name] = df[name].tolist()[: health["window_len"]]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Signal")
    if sensors:
        chart_df = pd.DataFrame(sensors)
        st.line_chart(chart_df)
    else:
        st.info("Provide a window (synthetic demo or CSV upload) to see it here.")

with col2:
    st.subheader("Prediction")
    if sensors and st.button("Predict", type="primary"):
        try:
            resp = requests.post(f"{API_URL}/predict", json={"sensors": sensors}, timeout=30)
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.metric("Predicted class", result["predicted_class"])
            proba_df = pd.DataFrame(
                {"class": list(result["probabilities"]), "probability": list(result["probabilities"].values())}
            ).sort_values("probability", ascending=False)
            st.bar_chart(proba_df.set_index("class"))

            st.write(f"**Conformal prediction set** (the classes the model cannot statistically "
                     f"rule out at the configured coverage level): {result['conformal_set']}")
            for w in result["warnings"]:
                st.warning(w)

            try:
                explain_resp = requests.post(f"{API_URL}/explain", json={"sensors": sensors}, timeout=30)
                explain_resp.raise_for_status()
                explanation = explain_resp.json()
                st.write(f"**Top contributing features** (method: {explanation['method']}):")
                st.dataframe(pd.DataFrame(explanation["top_features"]))
            except requests.RequestException:
                st.caption("Explanation unavailable.")

st.divider()
st.subheader("Data-quality audit")
if st.button("Run audit"):
    try:
        audit_resp = requests.get(f"{API_URL}/audit", timeout=120)
        audit_resp.raise_for_status()
        audit = audit_resp.json()
    except requests.RequestException as exc:
        st.error(f"Audit failed: {exc}")
    else:
        st.write(f"Windows analyzed: {audit['n_windows']}")
        st.write(f"Class balance: {audit['class_balance_verdict']}")
        st.write(f"Recommended sensors: {audit['recommended_sensors']}")
        st.write("Excluded sensors:")
        st.json(audit["excluded_sensors"])
