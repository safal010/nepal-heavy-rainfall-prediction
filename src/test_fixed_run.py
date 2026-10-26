
import requests
import pandas as pd

API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"

# Kathmandu
LATITUDE = 27.7172
LONGITUDE = 85.3240

# Predict rainfall for September 2 using the previous day's
# 00:00 UTC model initialization.
TARGET_DATE = "2025-09-02"
RUN = "2025-09-01T00:00"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
    ],
    "timezone": "Asia/Kathmandu",
    "models": "ecmwf_ifs",
    "run": RUN,
    "forecast_hours": 72,
}

print("Requesting fixed forecast run...")
print("Model initialization (UTC):", RUN)
print("Target date (Nepal local date):", TARGET_DATE)

response = requests.get(API_URL, params=params, timeout=60)

print("HTTP status:", response.status_code)

response.raise_for_status()
data = response.json()

if "hourly" not in data:
    raise RuntimeError(f"No hourly data returned: {data}")

hourly = pd.DataFrame(data["hourly"])
hourly["time"] = pd.to_datetime(hourly["time"])

print("\nFirst five returned timestamps:")
print(hourly["time"].head().to_string(index=False))

print("\nLast five returned timestamps:")
print(hourly["time"].tail().to_string(index=False))

# Select the target date using Nepal's local calendar date.
target = hourly[
    hourly["time"].dt.strftime("%Y-%m-%d") == TARGET_DATE
].copy()

print("\nHourly records for target date:", len(target))

variables = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

if len(target) != 24:
    raise ValueError(
        f"Expected 24 hourly records for {TARGET_DATE}, "
        f"but received {len(target)}."
    )

print("\nValid values for each variable:")
print(target[variables].count().to_string())

if target[variables].isna().any().any():
    raise ValueError("Some target-day forecast values are missing.")

print("\n--- FIXED-RUN FORECAST SUMMARY ---")
print("Target date:", TARGET_DATE)
print("Forecast rainfall (mm):", round(target["precipitation"].sum(), 3))
print("\nHourly rainfall for target date:")
print(
    target[["time", "precipitation"]]
    .to_string(index=False)
)

print("\nHours with at least 1 mm rainfall:")
print(
    target[target["precipitation"] >= 1]
    [["time", "precipitation"]]
    .to_string(index=False)
)
print("Mean temperature (°C):", round(target["temperature_2m"].mean(), 2))
print("Mean humidity (%):", round(target["relative_humidity_2m"].mean(), 2))
print("Mean wind speed (km/h):", round(target["wind_speed_10m"].mean(), 2))

print("\nTest completed successfully.")
