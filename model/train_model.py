import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing.data_loader import load_time_series
from preprocessing.scaler import scale_series
from preprocessing.sequence_builder import create_sequences
from model.lstm_autoencoder import build_lstm_autoencoder

if __name__ == "__main__":
    # Load and preprocess
    df = load_time_series("data/raw/weather_historical.csv")
    scaled_values, scaler = scale_series(df["value"].values)

    WINDOW_SIZE = 24
    X = create_sequences(scaled_values, WINDOW_SIZE)

    # Build model
    model = build_lstm_autoencoder(
        timesteps=X.shape[1],
        features=X.shape[2]
    )

    # Train
    history = model.fit(
        X, X,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        shuffle=False
    )

    # Save model
    model.save("model/saved_models/lstm_autoencoder.keras")

    print("✅ Model trained and saved successfully")
