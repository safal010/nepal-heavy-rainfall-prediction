
import requests
import pandas as pd

url = "https://single-runs-api.open-meteo.com/v1/forecast"

params = {
    "latitude": 27.7172,
    "longitude": 85.3240,
    "run": "2025-09-01T00:00",
    "models": "ecmwf_ifs",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
    ],
    "forecast_days": 3,
    "timezone": "Asia/Kathmandu",
}

response = requests.get(url, params=params, timeout=60)

print("HTTP status:", response.status_code)
print("API error details:", response.text)

response.raise_for_status()

data = response.json()

print("\n--- FORECAST RUN ---")
print("Requested run (UTC):", params["run"])
print("Model:", params["models"])
print("Timezone:", data.get("timezone"))
print("Returned coordinates:", data.get("latitude"), data.get("longitude"))

hourly = pd.DataFrame(data["hourly"])
hourly["time"] = pd.to_datetime(hourly["time"])

print("\n--- HOURLY COVERAGE ---")
print("Number of hourly records:", len(hourly))
print("First timestamp:", hourly["time"].min())
print("Last timestamp:", hourly["time"].max())

print("\n--- DAILY FORECAST ---")
hourly["date"] = hourly["time"].dt.date

daily = hourly.groupby("date").agg(
    hourly_records=("time", "size"),
    temperature_mean_c=("temperature_2m", "mean"),
    humidity_mean_pct=("relative_humidity_2m", "mean"),
    forecast_rainfall_mm=("precipitation", "sum"),
    wind_speed_mean_kmh=("wind_speed_10m", "mean"),
)

print(daily.to_string())


print("\n--- DAILY FORECAST COVERAGE ---")

hourly["date"] = hourly["time"].dt.date

daily = hourly.groupby("date").agg(
    hourly_records=("time", "size"),
    valid_temperature=("temperature_2m", "count"),
    valid_humidity=("relative_humidity_2m", "count"),
    valid_precipitation=("precipitation", "count"),
    valid_wind=("wind_speed_10m", "count"),
    temperature_mean_c=("temperature_2m", "mean"),
    humidity_mean_pct=("relative_humidity_2m", "mean"),
    forecast_rainfall_mm=("precipitation", "sum"),
    wind_speed_mean_kmh=("wind_speed_10m", "mean"),
)

daily["complete_day"] = (
    daily["hourly_records"].eq(24)
    & daily["valid_temperature"].eq(24)
    & daily["valid_humidity"].eq(24)
    & daily["valid_precipitation"].eq(24)
    & daily["valid_wind"].eq(24)
)

print(daily.to_string())

print("\n--- COMPLETE FORECAST DAYS ---")
print(daily[daily["complete_day"]].to_string())

print(hourly.isna().sum().to_string())
