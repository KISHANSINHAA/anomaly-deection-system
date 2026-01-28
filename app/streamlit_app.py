import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from preprocessing.data_loader import load_time_series
from preprocessing.scaler import scale_series
from preprocessing.sequence_builder import create_sequences
from anomaly.detector import reconstruction_error
from anomaly.injector import inject_spike, inject_freeze
from ingestion.live_weather_source import LiveWeatherSource


# --------------------------------------------------
# Utility
# --------------------------------------------------
def calculate_threshold(errors, percentile=99.5):
    return np.percentile(errors, percentile)


np.random.seed(42)  # stable demo


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="SentinelGuard – Anomaly Detection System",
    layout="wide"
)

st.title("🛡️ SentinelGuard – Anomaly Detection System")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Data Source")

mode = st.sidebar.radio(
    "Select mode",
    ["Historical (Training)", "Live (Inference)"]
)

if mode == "Historical (Training)":
    inject = st.sidebar.checkbox("Inject Synthetic Anomalies")
    simulate_live = False
else:
    inject = False
    simulate_live = st.sidebar.checkbox("🎭 Simulate Live Anomaly")
    st.sidebar.info("Injection available only in Historical mode")


# --------------------------------------------------
# Load model (cached)
# --------------------------------------------------
@st.cache_resource
def load_trained_model():
    return load_model("model/saved_models/lstm_autoencoder.keras")


model = load_trained_model()


# ==================================================
# HISTORICAL MODE
# ==================================================
if mode == "Historical (Training)":

    df = load_time_series("data/raw/weather_historical.csv")
    values = df["value"].values

    scaled, scaler = scale_series(values)
    WINDOW_SIZE = 24

    # -------- Baseline (clean) ----------
    X_clean = create_sequences(scaled, WINDOW_SIZE)
    X_clean_hat = model.predict(X_clean, verbose=0)
    errors_clean = reconstruction_error(X_clean, X_clean_hat)
    threshold = calculate_threshold(errors_clean, 99.5)

    # -------- Injection ----------
    if inject:
        scaled, _ = inject_spike(scaled, spike_value=1.2, count=25)
        scaled, _ = inject_freeze(scaled, length=72)

    values_plot = scaler.inverse_transform(
        scaled.reshape(-1, 1)
    ).flatten()

    # -------- Test ----------
    X_test = create_sequences(scaled, WINDOW_SIZE)
    X_hat = model.predict(X_test, verbose=0)
    errors_test = reconstruction_error(X_test, X_hat)
    anomaly_flags = errors_test > threshold

    # -------- Plot: temperature ----------
    st.subheader("📈 Temperature with Detected Anomalies")

    fig1, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(values_plot[WINDOW_SIZE:], label="Temperature")

    ax1.scatter(
        np.where(anomaly_flags)[0],
        values_plot[WINDOW_SIZE:][anomaly_flags],
        color="red",
        s=25,
        label="Detected Anomaly"
    )

    ax1.set_xlabel("Time Index")
    ax1.set_ylabel("Temperature")
    ax1.legend()
    st.pyplot(fig1)

    # -------- Plot: error ----------
    st.subheader("📉 Reconstruction Error Over Time")

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(errors_test, label="Reconstruction Error")
    ax2.axhline(threshold, color="red", linestyle="--", label="Baseline Threshold")
    ax2.set_xlabel("Sequence Index")
    ax2.set_ylabel("Error")
    ax2.legend()
    st.pyplot(fig2)

    # -------- Summary ----------
    st.subheader("📊 Detection Summary")
    st.write(f"**Baseline Threshold (99.5 percentile):** {threshold:.6f}")
    st.write(f"**Anomalies detected:** {int(anomaly_flags.sum())}")


# ==================================================
# LIVE MODE
# ==================================================
else:
    st.subheader("📡 Live Temperature Stream")

    source = LiveWeatherSource()

    if "live_buffer" not in st.session_state:
        st.session_state.live_buffer = []

    if "live_error_buffer" not in st.session_state:
        st.session_state.live_error_buffer = []

    if st.button("Fetch Next Data Point"):
        try:
            point = source.fetch_latest_temperature()
            value = point["value"]

            # 🎭 simulate anomaly
            if simulate_live:
                value = value + np.random.uniform(10, 15)

            st.session_state.live_buffer.append(value)

            if len(st.session_state.live_buffer) > 24:
                st.session_state.live_buffer.pop(0)

            st.write(f"🕒 {point['timestamp']}")
            st.write(f"🌡️ Temperature: **{value:.1f} °C**")

        except Exception as e:
            st.error(f"Live fetch failed: {e}")
            st.stop()

    if len(st.session_state.live_buffer) < 24:
        st.warning("Waiting for 24 data points to start inference...")
    else:
        buffer = np.array(st.session_state.live_buffer)

        # Use historical scaler & threshold
        df_hist = load_time_series("data/raw/weather_historical.csv")
        hist_scaled, scaler = scale_series(df_hist["value"].values)

        X_clean = create_sequences(hist_scaled, 24)
        X_clean_hat = model.predict(X_clean, verbose=0)
        threshold = calculate_threshold(
            reconstruction_error(X_clean, X_clean_hat),
            99.5
        )

        buffer_scaled = scaler.transform(buffer.reshape(-1, 1))
        X_live = buffer_scaled.reshape(1, 24, 1)
        X_hat = model.predict(X_live, verbose=0)

        error = np.mean(np.abs(X_live - X_hat))
        st.session_state.live_error_buffer.append(error)

        if len(st.session_state.live_error_buffer) > 50:
            st.session_state.live_error_buffer.pop(0)

        st.subheader("📊 Live Inference Result")
        st.write(f"Reconstruction Error: `{error:.6f}`")
        st.write(f"Baseline Threshold: `{threshold:.6f}`")

        if error > threshold:
            st.error("🚨 ANOMALY DETECTED")
        else:
            st.success("✅ Normal behavior")

        # -------- Rolling graph ----------
        st.subheader("📈 Live Rolling Graph")

        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        ax[0].plot(st.session_state.live_buffer, marker="o")
        ax[0].set_ylabel("Temperature (°C)")
        ax[0].set_title("Last 24 Temperature Points")

        ax[1].plot(st.session_state.live_error_buffer, color="orange")
        ax[1].axhline(threshold, color="red", linestyle="--", label="Threshold")
        ax[1].set_ylabel("Reconstruction Error")
        ax[1].set_xlabel("Time Step")
        ax[1].legend()

        st.pyplot(fig)
