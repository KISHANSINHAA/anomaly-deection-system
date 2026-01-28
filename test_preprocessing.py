import numpy as np
from preprocessing.data_loader import load_time_series
from preprocessing.scaler import scale_series
from preprocessing.sequence_builder import create_sequences

if __name__ == "__main__":
    df = load_time_series("data/raw/weather_historical.csv")

    scaled_values, scaler = scale_series(df["value"].values)

    WINDOW_SIZE = 24
    X = create_sequences(scaled_values, WINDOW_SIZE)

    print("Raw rows:", len(df))
    print("Scaled shape:", scaled_values.shape)
    print("Sequences shape:", X.shape)
