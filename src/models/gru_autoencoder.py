#!/usr/bin/env python3
"""
GRU Autoencoder (Secondary Model)
Single-file version: model + training + saving
LSTM remains primary model
"""

import os
import numpy as np
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# CONFIG (MATCH LSTM)
# =========================
SEQUENCE_LENGTH = 10
N_FEATURES = 1

X_TRAIN_PATH = "data/processed/finance/sequences/X_train.npy"
X_TEST_PATH  = "data/processed/finance/sequences/X_test.npy"

MODEL_DIR = "models_saved/gru"
MODEL_PATH = os.path.join(MODEL_DIR, "gru_autoencoder.keras")

EPOCHS = 40
BATCH_SIZE = 32


# =========================
# MODEL
# =========================
def build_gru_autoencoder():
    inputs = Input(shape=(SEQUENCE_LENGTH, N_FEATURES))

    # Encoder
    x = GRU(64, return_sequences=True, activation="tanh")(inputs)
    x = GRU(32, return_sequences=False, activation="tanh")(x)

    # Bottleneck
    x = RepeatVector(SEQUENCE_LENGTH)(x)

    # Decoder
    x = GRU(32, return_sequences=True, activation="tanh")(x)
    x = GRU(64, return_sequences=True, activation="tanh")(x)

    outputs = TimeDistributed(Dense(N_FEATURES))(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse"
    )

    return model


# =========================
# TRAINING
# =========================
def main():
    print("\n🚀 Training GRU Autoencoder (Secondary Model)")
    print("=" * 50)

    # ---- Load data ----
    if not os.path.exists(X_TRAIN_PATH) or not os.path.exists(X_TEST_PATH):
        raise FileNotFoundError(
            "❌ Sequence files not found.\n"
            "Expected:\n"
            f"  {X_TRAIN_PATH}\n"
            f"  {X_TEST_PATH}"
        )

    X_train = np.load(X_TRAIN_PATH)
    X_test  = np.load(X_TEST_PATH)

    logger.info(f"Train sequences: {X_train.shape}")
    logger.info(f"Test sequences : {X_test.shape}")

    # ---- Build model ----
    model = build_gru_autoencoder()
    model.summary()

    # ---- Train ----
    model.fit(
        X_train,
        X_train,
        validation_data=(X_test, X_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1
    )

    # ---- Save model ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    print("\n✅ GRU TRAINING COMPLETE")
    print(f"💾 Model saved at: {MODEL_PATH}")
    print("=" * 50)


    # ---- Save reconstruction errors (for UI & evaluation) ----
    print("\n📊 Generating GRU reconstruction errors...")
    reconstructed = model.predict(X_test, verbose=0)
    errors = np.mean(np.square(X_test - reconstructed), axis=(1, 2))

    error_dir = "data/processed/finance"
    os.makedirs(error_dir, exist_ok=True)

    np.save(os.path.join(error_dir, "gru_reconstruction_errors.npy"), errors)

    print("💾 Saved GRU reconstruction errors → data/processed/finance/gru_reconstruction_errors.npy")

if __name__ == "__main__":
    main()
