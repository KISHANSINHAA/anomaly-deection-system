import os
import numpy as np

# =====================================================
# TUNED CONFIG (ACCURACY-ORIENTED)
# =====================================================
PERCENTILE = 85            # ↑ threshold → fewer false positives
WINDOW_SIZE = 60
MIN_CONSECUTIVE = 4        # stronger temporal confirmation
ERROR_FLOOR_PERCENTILE = 70  # suppress weak noise


# =====================================================
# DYNAMIC THRESHOLD FUNCTION
# =====================================================
def dynamic_threshold(errors, percentile=85, window=60):
    thresholds = np.zeros(len(errors))
    for i in range(len(errors)):
        start = max(0, i - window)
        thresholds[i] = np.percentile(errors[start:i + 1], percentile)
    return thresholds


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":

    print("Running point-level evaluation...")

    error_path = "data/processed/finance/lstm_reconstruction_errors.npy"
    label_path = "data/processed/finance/true_labels.npy"

    if not os.path.exists(error_path):
        print("❌ Reconstruction errors not found")
        exit()

    errors = np.load(error_path)

    print(
        f"Error stats → min: {errors.min():.6f}, "
        f"max: {errors.max():.6f}, "
        f"mean: {errors.mean():.6f}"
    )

    # -------------------------------------------------
    # 1️⃣ Noise suppression (VERY IMPORTANT)
    # -------------------------------------------------
    error_floor = np.percentile(errors, ERROR_FLOOR_PERCENTILE)
    errors = np.where(errors < error_floor, 0.0, errors)

    # -------------------------------------------------
    # 2️⃣ Dynamic thresholding
    # -------------------------------------------------
    thresholds = dynamic_threshold(errors, PERCENTILE, WINDOW_SIZE)
    raw_preds = (errors > thresholds).astype(int)

    # -------------------------------------------------
    # 3️⃣ Strong temporal smoothing
    # -------------------------------------------------
    smoothed_preds = np.zeros_like(raw_preds)
    for i in range(len(raw_preds)):
        start = max(0, i - MIN_CONSECUTIVE + 1)
        if raw_preds[start:i + 1].mean() >= 0.75:
            smoothed_preds[i] = 1

    anomaly_count = int(smoothed_preds.sum())
    print("Total anomalies detected:", anomaly_count)

    # -------------------------------------------------
    # 4️⃣ Accuracy (proxy)
    # -------------------------------------------------
    if os.path.exists(label_path):
        y_true = np.load(label_path)
        min_len = min(len(y_true), len(smoothed_preds))
        accuracy = (y_true[:min_len] == smoothed_preds[:min_len]).mean()
        print(f"Detection Accuracy (proxy): {accuracy:.4f}")
    else:
        print("⚠️ true_labels.npy not found → accuracy skipped")
