import pandas as pd

def load_time_series(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Sort to be safe
    df = df.sort_values("timestamp")

    # Fill missing values (important)
    df["value"] = df["value"].interpolate(method="linear")

    return df
