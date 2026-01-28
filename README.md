# 🛡️ SentinelGuard – Anomaly Detection System

SentinelGuard is an **end-to-end anomaly detection system** built using an **LSTM Autoencoder** for **time-series data**.

It supports **historical anomaly detection**, **synthetic anomaly simulation**, and **live streaming inference** through an interactive **Streamlit dashboard**.

This project is designed to be **production-oriented**, **interview-ready**, and **deployment-friendly**.

---

## 🚀 Why SentinelGuard?

Modern systems generate massive time-series data (weather, sensors, finance, IoT).  
Traditional rule-based monitoring fails to capture **unknown or evolving anomalies**.

SentinelGuard solves this by learning **normal behavior only** and detecting deviations using **reconstruction error**.

---

## ✅ Key Design Goals

- ✅ Production-oriented architecture
- ✅ CDAC / ML interview defendable
- ✅ Clean modular Python code
- ✅ Streamlit Cloud compatible
- ✅ Real-time inference ready
- ✅ No labeled anomaly data required

---

## 🧠 Core Technology

- **Model**: LSTM Autoencoder  
- **Learning Type**: Unsupervised  
- **Input**: Time-series sequences (window size = 24)  
- **Detection Metric**: Reconstruction Error  
- **Threshold**: Learned from clean historical data (99.5 percentile)

---

## 🔍 Supported Use Cases

- 🌦️ Weather anomaly detection  
- 🧪 Sensor fault detection  
- 📈 Finance / stock anomalies  
- 📡 IoT telemetry monitoring  
- 🖥️ System health monitoring  

---

## ✨ Features

### 📊 Historical Anomaly Detection
- Trains on **clean historical data**
- Learns baseline behavior
- Detects anomalies using reconstruction error

### 🎭 Synthetic Anomaly Injection
- Injects artificial spikes & drops
- Used for **testing & validation**
- Disabled in live mode to avoid contamination

### 📡 Live Streaming Inference
- Fetches real-time temperature data
- Uses rolling window inference
- Detects anomalies **without retraining**
- Threshold reused from historical training

### 📈 Visualizations
- Time-series temperature plot
- Reconstruction error over time
- Anomaly markers
- Threshold visualization

---

## 🗂️ Project Structure

```text
anomaly-deection-system/
│
├── app/
│   └── streamlit_app.py          # Streamlit dashboard entry point
│
├── preprocessing/
│   ├── data_loader.py            # Load time-series data
│   ├── scaler.py                 # MinMax scaling
│   └── sequence_builder.py       # Sliding window creation
│
├── ingestion/
│   ├── historical_loader.py      # Historical dataset loader
│   └── live_weather_source.py    # Live temperature API source
│
├── model/
│   ├── train_model.py            # LSTM Autoencoder training
│   └── saved_models/
│       └── lstm_autoencoder.keras
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_historical_loader.py
│   ├── test_anomaly_detection.py
│   └── test_weather.py
│
├── requirements.txt
├── README.md
└── .gitignore
