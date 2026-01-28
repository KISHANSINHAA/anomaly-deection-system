import requests
import pandas as pd
from datetime import datetime, timedelta

class HistoricalWeatherLoader:

    def fetch_historical_temperature_chunked(
        self,
        latitude,
        longitude,
        start_date,
        end_date,
        save_path="data/raw/weather_historical.csv"
    ):
        url = "https://archive-api.open-meteo.com/v1/archive"

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_dfs = []

        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=30), end)

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": current.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d"),
                "hourly": "temperature_2m",
                "timezone": "auto"
            }

            print(f"Fetching: {params['start_date']} → {params['end_date']}")

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame({
                "timestamp": data["hourly"]["time"],
                "value": data["hourly"]["temperature_2m"]
            })

            all_dfs.append(df)
            current = chunk_end + timedelta(days=1)

        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df["timestamp"] = pd.to_datetime(final_df["timestamp"])

        final_df.to_csv(save_path, index=False)
        return final_df
