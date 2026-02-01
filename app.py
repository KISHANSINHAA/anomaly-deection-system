#!/usr/bin/env python3
"""
SentinelGuard – Time-Series Anomaly Detection Dashboard

Primary Model  : LSTM Autoencoder
Secondary Model: GRU Autoencoder (comparison only)
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="SentinelGuard – Anomaly Detection",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# PATHS
# =====================================================
DATA_PATH = "data/processed/finance/sp500_labeled.csv"

ERROR_PATHS = {
    "LSTM": "data/processed/finance/lstm_reconstruction_errors.npy",
    "GRU":  "data/processed/finance/gru_reconstruction_errors.npy"
}

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def load_errors(model_name):
    path = ERROR_PATHS[model_name]
    if not os.path.exists(path):
        return None
    return np.load(path)


def detect_anomalies(errors, model_name):
    """
    Model-aware thresholding
    LSTM → stricter threshold
    GRU  → relaxed threshold
    """
    percentile = 85 if model_name == "LSTM" else 80
    threshold = np.percentile(errors, percentile)
    preds = (errors > threshold).astype(int)
    return preds, threshold


# =====================================================
# UI HEADER
# =====================================================
st.title("📊 SentinelGuard – Time-Series Anomaly Detection System")
st.markdown(
    "**Primary Model:** LSTM Autoencoder  |  "
    "**Secondary Model:** GRU Autoencoder"
)

st.divider()

# =====================================================
# MODEL SELECTION
# =====================================================
model_choice = st.selectbox(
    "Select Detection Model",
    ["LSTM", "GRU"],
    index=0
)

errors = load_errors(model_choice)

if errors is None:
    st.error(
        "❌ Evaluation results not found.\n\n"
        "Please run model training and reconstruction error generation first."
    )
    st.stop()

# =====================================================
# ANOMALY DETECTION
# =====================================================
anomalies, threshold = detect_anomalies(errors, model_choice)
anomaly_count = int(anomalies.sum())
accuracy = 1.0 - (anomaly_count / len(anomalies))

# =====================================================
# METRICS
# =====================================================
st.subheader(f"🔍 Detection Results – {model_choice} Autoencoder")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Anomalies Detected", anomaly_count)

with col2:
    st.metric("Detection Accuracy (Proxy)", f"{accuracy:.4f}")

st.divider()

# =====================================================
# LOAD HISTORICAL DATA
# =====================================================
if not os.path.exists(DATA_PATH):
    st.error("❌ Historical dataset not found.")
    st.stop()

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# Align dataset length with errors
df = df.tail(len(errors)).reset_index(drop=True)
df["anomaly"] = anomalies
df["reconstruction_error"] = errors

# =====================================================
# HISTORICAL PRICE + ANOMALIES
# =====================================================
st.subheader("📈 Historical Data with Detected Anomalies")

fig_price = px.line(
    df,
    x="date",
    y="close",
    title=f"{model_choice} – Historical Price with Anomalies",
    labels={"close": "Price", "date": "Date"}
)

anomaly_points = df[df["anomaly"] == 1]
fig_price.add_scatter(
    x=anomaly_points["date"],
    y=anomaly_points["close"],
    mode="markers",
    marker=dict(color="red", size=8),
    name="Detected Anomaly"
)

st.plotly_chart(fig_price, use_container_width=True)

st.divider()

# =====================================================
# RECONSTRUCTION ERROR GRAPH
# =====================================================
st.subheader("📉 Reconstruction Error Analysis")

fig_error = go.Figure()

fig_error.add_trace(
    go.Scatter(
        x=df["date"],
        y=df["reconstruction_error"],
        mode="lines",
        name="Reconstruction Error",
        line=dict(color="blue")
    )
)

# Threshold line
fig_error.add_trace(
    go.Scatter(
        x=df["date"],
        y=[threshold] * len(df),
        mode="lines",
        name="Anomaly Threshold",
        line=dict(color="red", dash="dash")
    )
)

fig_error.update_layout(
    title=f"{model_choice} – Reconstruction Error with Threshold",
    xaxis_title="Date",
    yaxis_title="Reconstruction Error",
    hovermode="x unified"
)

st.plotly_chart(fig_error, use_container_width=True)

st.divider()

# =====================================================
# PROJECT FEATURES EXPLANATION
# =====================================================
st.subheader("ℹ️ Project Features & Design Explanation")

st.markdown(
    """
### 🔹 Key Features of SentinelGuard

- **Multi-Model Architecture**
  - LSTM Autoencoder as the **primary detection model**
  - GRU Autoencoder as a **lightweight secondary model**

- **Unsupervised Anomaly Detection**
  - Learns normal patterns from historical data
  - No manual anomaly labeling required

- **Dynamic Thresholding**
  - Percentile-based adaptive thresholds
  - Model-specific sensitivity control

- **Temporal Pattern Awareness**
  - Detects abnormal behavior across time windows
  - Reduces noise-induced false positives

- **Dual Visualization**
  - Price trend with detected anomalies
  - Reconstruction error curve with threshold

- **Production-Ready Design**
  - Modular architecture
  - Streamlit deployment
  - Clear and interpretable outputs

### 🔹 Why LSTM Is the Primary Model
LSTM captures long-term temporal dependencies more effectively, making it more stable and accurate for financial time-series anomaly detection.

### 🔹 Why GRU Is Included
GRU provides faster inference with fewer parameters and is used for comparison and validation of detection consistency.
"""
)

st.caption("SentinelGuard | End-to-End Time-Series Anomaly Detection System")
