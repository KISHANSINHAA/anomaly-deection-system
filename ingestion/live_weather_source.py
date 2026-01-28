import requests
from datetime import datetime

class LiveWeatherSource:
    def __init__(self, latitude=28.61, longitude=77.23):
        self.lat = latitude
        self.lon = longitude

    def fetch_latest_temperature(self):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": "temperature_2m",
            "forecast_days": 1
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                raise RuntimeError("API returned non-200 status")

            if not response.text.strip():
                raise RuntimeError("Empty API response")

            data = response.json()

            temp = data["hourly"]["temperature_2m"][-1]
            time = data["hourly"]["time"][-1]

            return {
                "timestamp": datetime.fromisoformat(time),
                "value": float(temp)
            }

        except Exception as e:
            raise RuntimeError(f"Live API failure: {e}")
