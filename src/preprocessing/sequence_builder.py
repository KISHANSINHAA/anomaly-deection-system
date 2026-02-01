#!/usr/bin/env python3
"""
Sequence builder for time-series anomaly detection
Creates leakage-free train/test datasets for LSTM/GRU
"""

import pandas as pd
import numpy as np
import os
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_PATH = "data/processed/finance/sp500_labeled.csv"
OUTPUT_DIR = "data/processed/finance/sequences"

SEQUENCE_LENGTH = 10
TRAIN_RATIO = 0.7


def build_sequences(values, labels, seq_len):
    X, y = [], []

    for i in range(len(values) - seq_len):
        X.append(values[i:i + seq_len])
        y.append(labels[i + seq_len])  # label = last point

    return np.array(X), np.array(y)


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    df = df.sort_values("date").reset_index(drop=True)

    # Use log_return only (clean signal)
    values = df["log_return"].values.reshape(-1, 1)
    labels = df["is_anomaly"].values

    # Build sequences
    X, y = build_sequences(values, labels, SEQUENCE_LENGTH)

    # Time-based split (NO shuffle)
    split_idx = int(len(X) * TRAIN_RATIO)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    logger.info("Sequence building completed")
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Anomaly rate (train): {y_train.mean():.2%}")
    logger.info(f"Anomaly rate (test): {y_test.mean():.2%}")

    print("✅ Sequence data ready for model training")


if __name__ == "__main__":
    main()
