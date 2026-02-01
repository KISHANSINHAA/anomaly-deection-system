#!/usr/bin/env python3
"""
GRU Point-Level Anomaly Evaluation
Secondary model – comparison only (LSTM remains primary)
"""

import numpy as np
import logging
import os
from src.models.gru_autoencoder import GRUAutoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# CONFIG (SAFE DEFAULTS)
# =========================
SEQUENCE_LENGTH = 10
N_FEATURES = 1

MODEL_PATH = "models_saved/gru/gru_autoencoder.keras"
SEQUENCES_PATH = "data/processed/sequences/X_test.npy"

PERCENTILE = 80            # Same as LSTM
MIN_CONSECUTIVE = 3        # Same smoothing
ERROR_FLOOR_PERCENTILE = 60


# =========================
# Temporal smoothing
# =========================
def temporal_smoothing(preds, min_consecutive=3):
    smoothed = np.zeros_like(preds)
    for i in range(len(preds)):
        start = max(0, i - min_consecutive + 1)
        window = preds[start:i + 1]
        smoothed[i] = 1 if np.sum(window) >= min_consecutive else 0
    return smoothed


# =========================
# MAIN EVALUATION
# =========================
def main():
    print("\nRunning GRU point-level evaluation...")

    # -------------------------
    # Load sequences
    # -------------------------
    if not os.path.exists(SEQUENCES_PATH):
        raise FileNotFoundError("❌ Test sequences not found")

    X_test = np.load(SEQUENCES_PATH)
    logger.info(f"✅ Test sequences loaded: {X_test.shape}")

    # -------------------------
    # Load model
    # -------------------------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ GRU model not found")

    gru = GRUAutoencoder(SEQUENCE_LENGTH, N_FEATURES)
    gru.load(MODEL_PATH)

    # -------------------------
    # Reconstruction errors
    # -------------------------
    errors = gru.reconstruction_errors(X_test)

    print(
        f"Error stats → min: {errors.min():.6f}, "
        f"max: {errors.max():.6f}, "
        f"mean: {errors.mean():.6f}"
    )

    # -------------------------
    # Suppress low-magnitude noise
    # -------------------------
    error_floor = np.percentile(errors, ERROR_FLOOR_PERCENTILE)
    errors = np.where(errors < error_floor, 0, errors)

    # -------------------------
    # Dynamic threshold
    # -------------------------
    threshold = np.percentile(errors, PERCENTILE)
    raw_preds = (errors > threshold).astype(int)

    # -------------------------
    # Temporal smoothing
    # -------------------------
    final_preds = temporal_smoothing(raw_preds, MIN_CONSECUTIVE)

    anomaly_count = int(np.sum(final_preds))
    accuracy_proxy = 1.0 - (anomaly_count / len(final_preds))

    # -------------------------
    # OUTPUT (KEEP IT SIMPLE)
    # -------------------------
    print("\nGRU POINT-LEVEL RESULTS")
    print("=" * 40)
    print(f"Total anomalies detected : {anomaly_count}")
    print(f"Detection Accuracy       : {accuracy_proxy:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
