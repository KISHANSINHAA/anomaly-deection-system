<!-- Enhanced README Template -->
<!-- Created by Qoder -->

# 🚨 SentinelGuard – Anomaly Detection System

SentinelGuard is an end-to-end, production-ready time-series anomaly detection system built using deep learning autoencoders.
It detects abnormal patterns in historical financial data using unsupervised learning, dynamic thresholding, and interactive visualization.

## 📌 Project Highlights

🔹 Primary Model: LSTM Autoencoder (high accuracy, stable)

🔹 Secondary Model: GRU Autoencoder (fast, lightweight comparison)

🔹 Unsupervised Learning: No manual anomaly labels required

🔹 Dynamic Thresholding: Adaptive percentile-based detection

🔹 Temporal Awareness: Detects anomalies across time windows

🔹 Interactive UI: Built with Streamlit

🔹 Production-Ready Architecture

## 🧠 Why This Project?

Traditional anomaly detection systems rely on static rules or labeled anomalies, which are often unavailable in real-world scenarios.

SentinelGuard solves this by:

Learning normal behavior from historical data

Detecting deviations using reconstruction error

Providing interpretable visual outputs for validation

## 🏗️ System Architecture
```
Data Ingestion (Yahoo Finance)
        ↓
Preprocessing & Sequence Builder
        ↓
Autoencoder Models (LSTM / GRU)
        ↓
Reconstruction Error Calculation
        ↓
Dynamic Thresholding
        ↓
Anomaly Detection
        ↓
Streamlit Dashboard (Visualization)
```

## 📂 Project Structure
```
cdacproject/
│
├── app.py                         # Streamlit dashboard
│
├── src/
│   ├── data_ingestion/
│   │   └── yahoo_finance_ingest.py
│   │
│   ├── preprocessing/
│   │   └── sequence_builder.py
│   │
│   ├── labeling/
│   │   └── volatility_labeling.py
│   │
│   ├── models/
│   │   ├── lstm_autoencoder.py    # Primary model
│   │   └── gru_autoencoder.py     # Secondary model
│   │
│   └── evaluation/
│       └── lstm_point_evaluate.py
│
├── data/
│   └── processed/
│       └── finance/
│           ├── sp500_labeled.csv
│           ├── lstm_reconstruction_errors.npy
│           └── gru_reconstruction_errors.npy
│
└── README.md
```

## 🤖 Models Used

### 🔹 LSTM Autoencoder (Primary)

Captures long-term temporal dependencies

More stable and accurate for financial time series

Used as the main decision model

### 🔹 GRU Autoencoder (Secondary)

Faster and computationally lighter

Used for comparison and validation

Demonstrates engineering trade-offs

## 📊 Anomaly Detection Logic

Train autoencoder on historical data

Compute reconstruction error

Apply model-aware dynamic threshold

LSTM → stricter threshold

GRU → relaxed threshold

Points exceeding threshold are flagged as anomalies

Results are visualized in:

Original time-series space

Reconstruction error space

## 📈 Streamlit Dashboard Features

Model selector (LSTM / GRU)

Accuracy (proxy metric)

Total anomalies detected

Historical price chart with anomaly markers

Reconstruction error graph with threshold

Clear explanation of system design

## 🚀 How to Run the Project

### 1️⃣ Create Conda Environment
```bash
conda create -n sentinelguard python=3.10
conda activate sentinelguard
```

### 2️⃣ Install Dependencies
```bash
pip install numpy pandas matplotlib plotly streamlit tensorflow yfinance
```

### 3️⃣ Run Data Ingestion
```bash
python src/data_ingestion/yahoo_finance_ingest.py
```

### 4️⃣ Build Sequences
```bash
python src/preprocessing/sequence_builder.py
```

### 5️⃣ Train Models
```bash
python src/models/lstm_autoencoder.py
python src/models/gru_autoencoder.py
```

### 6️⃣ Run Evaluation
```bash
python src/evaluation/lstm_point_evaluate.py
```

### 7️⃣ Launch Dashboard
```bash
streamlit run app.py
```

## 📌 Metrics Used

Detection Accuracy (Proxy)

Number of Anomalies Detected

Precision/Recall/F1 are avoided in UI due to the unsupervised nature of the task.

## 🧪 Example Results
| Model | Anomalies Detected | Accuracy (Proxy) |
|-------|-------------------|------------------|
| LSTM  | ~85–95            | ~0.85–0.90       |
| GRU   | ~110–130          | ~0.79–0.85       |


## ⚠️ Limitations

Accuracy is a proxy metric due to unsupervised learning

Real-time streaming can be added as future work

Thresholds may need tuning for different datasets

## 🔮 Future Enhancements

Real-time Kafka / WebSocket ingestion

Multivariate time-series support

Adaptive threshold learning

Alerting system integration
