import numpy as np
from tensorflow.keras.models import load_model

from preprocessing.data_loader import load_time_series
from preprocessing.scaler import scale_series
from preprocessing.sequence_builder import create_sequences
from anomaly.detector import reconstruction_error
from anomaly.threshold import calculate_threshold

if __name__ == "__main__":
    # Load model
    model = load_model("model/saved_models/lstm_autoencoder.keras")

    # Load data
    df = load_time_series("data/raw/weather_historical.csv")
    scaled_values, scaler = scale_series(df["value"].values)

    WINDOW_SIZE = 24
    X = create_sequences(scaled_values, WINDOW_SIZE)

    # Reconstruction
    X_hat = model.predict(X)

    errors = reconstruction_error(X, X_hat)
    threshold = calculate_threshold(errors)

    anomalies = errors > threshold

    print("Total sequences:", len(errors))
    print("Threshold:", threshold)
    print("Anomalies detected:", anomalies.sum())
