
import requests
import pandas as pd

# 1. Download the same archived forecast run
url = "https://single-runs-api.open-meteo.com/v1/forecast"

params = {
    "latitude": 27.7172,
    "longitude": 85.3240,
    "hourly": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "precipitation,"
        "wind_speed_10m"
    ),
    "timezone": "UTC",
    "models": "ecmwf_ifs",
    "run": "2025-09-01T00:00",
    "forecast_hours": 96,
}

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()
data = response.json()

# 2. Build a table from the hourly forecast
df = pd.DataFrame({
    "time": data["hourly"]["time"],
    "forecast_rainfall_mm": data["hourly"]["precipitation"],
})

df["time"] = pd.to_datetime(df["time"], utc=True)
df["date"] = df["time"].dt.date

# 3. Calculate rainfall totals for each UTC date
daily = df.groupby("date").agg(
    hourly_records=("time", "count"),
    valid_rainfall=("forecast_rainfall_mm", "count"),
    forecast_rainfall_mm=("forecast_rainfall_mm", "sum"),
)

print("\n--- FORECAST RAINFALL BY UTC DATE ---")
print(daily.to_string())

# 4. Load observed NASA POWER rainfall
observations = pd.read_csv("data/nepal_weather.csv")
observations["date"] = pd.to_datetime(
    observations["date"]
).dt.date

kathmandu = observations[
    observations["location"] == "Kathmandu"
].copy()

kathmandu = kathmandu[
    kathmandu["date"].isin([
        pd.Timestamp("2025-09-01").date(),
        pd.Timestamp("2025-09-02").date(),
        pd.Timestamp("2025-09-03").date(),
        pd.Timestamp("2025-09-04").date(),
    ])
]

print("\n--- NASA POWER UTC OBSERVATIONS ---")
print(
    kathmandu[["date", "PRECTOTCORR"]]
    .to_string(index=False)
)

print("\nCompare only complete forecast days.")

