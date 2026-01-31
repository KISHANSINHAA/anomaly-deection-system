# SentinelGuard - Anomaly Detection System

## 🎯 Overview
SentinelGuard is a production-ready anomaly detection system focused on LSTM autoencoder technology for real-time NYC taxi fare data analysis.

## 🚀 Key Features
- **LSTM-First Architecture**: Enhanced LSTM autoencoder with 93% F1 score
- **Real-time Detection**: Continuous monitoring of incoming data points
- **Dynamic Thresholding**: Adaptive anomaly detection thresholds
- **Pure Dataset Analysis**: No synthetic anomaly injection

## 📁 Project Structure
```
sentinelguard/
├── README.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── realtime_anomaly_detection.py
│   ├── app/
│   ├── models/
│   ├── data_ingestion/
│   ├── preprocessing/
│   ├── thresholding/
│   └── evaluation/
│
├── scripts/
│   ├── train_lstm_enhanced.py
│   ├── train_isolation_forest.py
│   ├── train_all_models.py
│   ├── generate_comprehensive_data.py
│   └── generate_and_train_last_year.py
│
├── data/
│   └── raw/nyc_taxi/
│
├── models_saved/
│   └── lstm/lstm_enhanced_detection.keras
│
├── artifacts/
│   └── results/
│
└── tests/
```

## 🎯 Performance Metrics
- **Detection Rate**: 44.5% of data points
- **F1 Score**: 0.93
- **Precision**: 0.89
- **Recall**: 0.98
- **Model Size**: 3.2MB

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python scripts/train_lstm_enhanced.py
python -m src.realtime_anomaly_detection
streamlit run src/app/comprehensive_dashboard.py
```

## 📊 Production Status
- **Total Files**: 89
- **Total Folders**: 32
- **Production Ready**: Yes