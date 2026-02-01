#!/usr/bin/env python3
"""
Volatility-based anomaly labeling for financial time series
Creates real ground-truth labels using rolling volatility
"""

import pandas as pd
import numpy as np
import os
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_PATH = "data/processed/finance/sp500_processed.csv"
OUTPUT_PATH = "data/processed/finance/sp500_labeled.csv"

ROLLING_WINDOW = 20        # ~1 trading month
STD_MULTIPLIER = 3.0       # Industry standard (2.5–4.0)


def label_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Applying volatility-based anomaly labeling")

    # Absolute log return
    df["abs_log_return"] = df["log_return"].abs()

    # Rolling statistics
    rolling_mean = df["abs_log_return"].rolling(ROLLING_WINDOW).mean()
    rolling_std = df["abs_log_return"].rolling(ROLLING_WINDOW).std()

    # Dynamic threshold
    df["volatility_threshold"] = rolling_mean + STD_MULTIPLIER * rolling_std

    # Label anomalies
    df["is_anomaly"] = (
        df["abs_log_return"] > df["volatility_threshold"]
    ).astype(int)

    # Drop early rows without rolling window
    df = df.dropna().reset_index(drop=True)

    return df


def save_labeled_data(df: pd.DataFrame):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved labeled data → {OUTPUT_PATH}")


if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Processed data not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])

    labeled_df = label_anomalies(df)

    anomaly_count = labeled_df["is_anomaly"].sum()
    anomaly_rate = anomaly_count / len(labeled_df)

    logger.info(f"Total samples: {len(labeled_df)}")
    logger.info(f"Anomalies detected: {anomaly_count}")
    logger.info(f"Anomaly rate: {anomaly_rate:.2%}")

    save_labeled_data(labeled_df)

    logger.info("✅ Volatility-based labeling completed successfully")
