.

🛡️ SentinelGuard – Anomaly Detection System

SentinelGuard is an end-to-end anomaly detection system built using an LSTM Autoencoder for time-series data.
It supports historical analysis, synthetic anomaly testing, and live streaming inference with a real-time dashboard.

This project is designed to be production-oriented, interview-defendable, and extensible to domains like:

Weather monitoring

Sensor fault detection

Finance / stock anomalies

IoT telemetry

System health monitoring

🚀 Key Features
✅ Historical Anomaly Detection

LSTM Autoencoder trained on historical time-series data

Baseline anomaly threshold learned from clean data only

Visualizations:

Temperature trends

Reconstruction error

Detected anomalies

🎭 Synthetic Anomaly Injection (Validation)

Injects realistic anomalies:

Sudden spikes

Sensor freeze (flat values)

Used only for testing, never for training

Demonstrates detection sensitivity and correctness

📡 Live Inference Mode

Fetches real-time temperature data using Open-Meteo API

Maintains a rolling window of 24 data points

Uses a fixed baseline threshold

No retraining, no threshold drift

📈 Live Rolling Visualization

Rolling temperature graph

Rolling reconstruction error graph

Threshold comparison in real time

🎭 Live Anomaly Simulation

Simulates sensor faults (sudden spikes)

Does not modify model or threshold

Useful for demos and interviews

🔁 CI/CD with GitHub Actions

Dependency installation checks

Environment consistency validation

Import and syntax verification

🧠 Architecture Overview
Historical Data
     ↓
Train LSTM Autoencoder
     ↓
Learn Baseline Threshold
     ↓
---------------------------------
     ↓
Live Stream → Scale → Window(24) → Model → Error → Threshold → Alert

🛠️ Tech Stack
Category	Technology
ML Model	LSTM Autoencoder (TensorFlow / Keras)
Backend	Python
Frontend	Streamlit
Visualization	Matplotlib
Data	Open-Meteo API + CSV
CI/CD	GitHub Actions
Environment	Conda
Version Control	Git
📁 Project Structure
sentinelguard/
│
├── app/
│   └── streamlit_app.py
│
├── ingestion/
│   ├── historical_weather_loader.py
│   └── live_weather_source.py
│
├── preprocessing/
│   ├── data_loader.py
│   ├── scaler.py
│   └── sequence_builder.py
│
├── anomaly/
│   ├── detector.py
│   └── injector.py
│
├── model/
│   ├── train_model.py
│   └── saved_models/
│
├── data/
│   └── raw/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── requirements.txt
└── README.md

🐍 Conda Environment Setup

This project is developed and tested using a Conda environment.

1️⃣ Create a new Conda environment
conda create -n sentinelguard python=3.10 -y

2️⃣ Activate the environment
conda activate sentinelguard

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Verify installation (optional)
python - <<EOF
import tensorflow
import streamlit
import numpy
print("Environment setup successful")
EOF

5️⃣ Deactivate environment (when done)
conda deactivate

▶️ Run the Application
streamlit run app/streamlit_app.py


Open in browser:

http://localhost:8501

🧪 How to Use
🔹 Historical Mode

Select Historical (Training)

Optionally enable Inject Synthetic Anomalies

Observe:

Reconstruction error

Detected anomalies

Summary metrics

🔹 Live Mode

Select Live (Inference)

Click Fetch Next Data Point

Wait for 24 data points (warm-up phase)

Observe real-time inference

Enable 🎭 Simulate Live Anomaly for demo

📊 Thresholding Strategy

Threshold computed using 99.5 percentile of reconstruction error

Learned only from clean historical data

Reused for:

Synthetic testing

Live inference

Prevents:

Data leakage

Adaptive masking

False positives

🎤 Interview-Ready Highlights

Strict separation of training and inference

No retraining during live monitoring

Robust API failure handling

Rolling window enforcement for LSTM

CI pipeline ensures reproducibility

🔮 Future Enhancements

Multivariate anomaly detection

Transformer-based models

Alerting (Email / Slack)

Docker & cloud deployment

Kafka-based streaming

👨‍💻 Contributors

Kishan – ML Engineer
(CDAC Project – SentinelGuard)

📜 License

This project is intended for educational and research purposes.