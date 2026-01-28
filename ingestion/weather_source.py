import requests
import pandas as pd

class WeatherSource:
    def fetch_hourly_temperature(self, latitude, longitude, past_days=7):
        """
        Fetch hourly temperature data from Open-Meteo API
        """
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m",
            "past_days": past_days
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame({
            "timestamp": data["hourly"]["time"],
            "value": data["hourly"]["temperature_2m"]
        })

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
