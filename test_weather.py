from ingestion.weather_source import WeatherSource

if __name__ == "__main__":
    source = WeatherSource()

    # Delhi coordinates (change anytime)
    df = source.fetch_hourly_temperature(
        latitude=28.61,
        longitude=77.23,
        past_days=7
    )

    print(df.head())
    print("\nRows:", len(df))
    print("Min temp:", df["value"].min())
    print("Max temp:", df["value"].max())
