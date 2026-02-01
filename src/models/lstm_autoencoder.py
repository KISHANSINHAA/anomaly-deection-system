#!/usr/bin/env python3
"""
LSTM Autoencoder - Primary Anomaly Detection Model
Trained ONLY on normal sequences
"""

import numpy as np
import os
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data/processed/finance/sequences"
MODEL_DIR = "models_saved/lstm"

SEQUENCE_LENGTH = 10
N_FEATURES = 1


def load_data():
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")

    # Train ONLY on clean normal data
    # Remove top 20% highest-variance sequences
    normal_idx = np.where(y_train == 0)[0]
    X_normal = X_train[normal_idx]

    # Variance per sequence
    seq_var = np.var(X_normal, axis=(1, 2))
    var_threshold = np.percentile(seq_var, 80)

    X_clean = X_normal[seq_var < var_threshold]

    logger.info(f"Normal sequences: {len(X_normal)}")
    logger.info(f"Clean normals used: {len(X_clean)}")

    return X_clean 



def build_lstm_autoencoder():
    inputs = Input(shape=(SEQUENCE_LENGTH, N_FEATURES))

    # Encoder
    x = LSTM(64, activation="tanh", return_sequences=True)(inputs)
    x = LSTM(32, activation="tanh", return_sequences=False)(x)

    # Bottleneck
    encoded = Dense(16, activation="tanh")(x)

    # Decoder
    x = RepeatVector(SEQUENCE_LENGTH)(encoded)
    x = LSTM(32, activation="tanh", return_sequences=True)(x)
    x = LSTM(64, activation="tanh", return_sequences=True)(x)

    outputs = TimeDistributed(Dense(N_FEATURES))(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse"
    )

    return model


def train():
    X_train = load_data()

    model = build_lstm_autoencoder()
    model.summary(print_fn=logger.info)

    callbacks = [
        EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    logger.info("Training LSTM autoencoder...")
    model.fit(
        X_train,
        X_train,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/lstm_autoencoder.keras")

    logger.info("✅ LSTM autoencoder trained and saved successfully")


if __name__ == "__main__":
    train()
