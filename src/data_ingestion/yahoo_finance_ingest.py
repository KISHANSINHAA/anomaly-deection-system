#!/usr/bin/env python3
"""
Yahoo Finance Data Ingestion
Downloads long-term S&P 500 data for anomaly detection
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import logging
import numpy as np

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOL = "^GSPC"
START_DATE = "2014-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

RAW_DIR = "data/raw/finance"
PROCESSED_DIR = "data/processed/finance"


def fetch_yahoo_data():
    logger.info(f"Fetching Yahoo Finance data: {SYMBOL}")
    logger.info(f"Date range: {START_DATE} → {END_DATE}")

    df = yf.download(
        SYMBOL,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        raise RuntimeError("No data fetched from Yahoo Finance")

    df.reset_index(inplace=True)
    return df


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles yfinance MultiIndex columns safely
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def clean_and_process(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning and processing data")

    # Fix MultiIndex columns
    df = flatten_columns(df)

    # Normalize column names
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    required_cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[required_cols].copy()

    # Drop missing rows
    df.dropna(inplace=True)

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Log returns (core signal for anomaly detection)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


def save_data(raw_df, processed_df):
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw_path = os.path.join(RAW_DIR, "sp500_raw.csv")
    processed_path = os.path.join(PROCESSED_DIR, "sp500_processed.csv")

    raw_df.to_csv(raw_path, index=False)
    processed_df.to_csv(processed_path, index=False)

    logger.info(f"Saved raw data → {raw_path}")
    logger.info(f"Saved processed data → {processed_path}")


if __name__ == "__main__":
    raw_df = fetch_yahoo_data()
    processed_df = clean_and_process(raw_df)
    save_data(raw_df, processed_df)

    logger.info("✅ Yahoo Finance ingestion completed successfully")
