from ingestion.historical_weather_loader import HistoricalWeatherLoader

if __name__ == "__main__":
    loader = HistoricalWeatherLoader()

    df = loader.fetch_historical_temperature_chunked(
        latitude=28.61,
        longitude=77.23,
        start_date="2023-01-01",
        end_date="2024-12-31",
        save_path="data/raw/weather_historical.csv"
    )

    print("\nDONE")
    print("Rows:", len(df))
    print("Min temp:", df["value"].min())
    print("Max temp:", df["value"].max())
