import os
import numpy as np
import pandas as pd
import logging
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================================================
# PATHS
# ==================================================
MODEL_PATH = "models_saved/lstm/lstm_autoencoder.keras"
SEQUENCE_PATH = "data/processed/finance/sequences/X_test.npy"
LABEL_PATH = "data/processed/finance/sequences/y_test.npy"

SAVE_DIR = "data/processed/finance"
ERROR_FILE = "lstm_reconstruction_errors.npy"
TRUE_LABEL_FILE = "true_labels.npy"


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    print("🔄 Generating LSTM Reconstruction Errors")

    if not os.path.exists(MODEL_PATH):
        print("❌ LSTM model not found:", MODEL_PATH)
        exit()

    if not os.path.exists(SEQUENCE_PATH):
        print("❌ Test sequences not found:", SEQUENCE_PATH)
        exit()

    # Load model
    model = load_model(MODEL_PATH)
    print("✅ LSTM model loaded")

    # Load test data
    X_test = np.load(SEQUENCE_PATH)
    print("✅ Test sequences loaded:", X_test.shape)

    # Predict
    print("🔍 Running reconstruction...")
    reconstructed = model.predict(X_test, verbose=0)

    # Compute reconstruction errors (MSE per sequence)
    errors = np.mean((X_test - reconstructed) ** 2, axis=(1, 2))
    print("📊 Error stats:",
          f"min={errors.min():.6f},",
          f"max={errors.max():.6f},",
          f"mean={errors.mean():.6f}")

    # Save errors
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(os.path.join(SAVE_DIR, ERROR_FILE), errors)
    print("💾 Saved reconstruction errors →", ERROR_FILE)

    # Save true labels if present
    if os.path.exists(LABEL_PATH):
        y_test = np.load(LABEL_PATH)
        np.save(os.path.join(SAVE_DIR, TRUE_LABEL_FILE), y_test)
        print("💾 Saved true labels →", TRUE_LABEL_FILE)

    print("✅ Reconstruction error generation complete")
