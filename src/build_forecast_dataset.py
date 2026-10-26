
import requests
import pandas as pd
import time
from pathlib import Path

# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OBSERVED_FILE = BASE_DIR / "data" / "nepal_weather.csv"
OUTPUT_FILE = BASE_DIR / "data" / "forecast_training_data.csv"

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"
HEAVY_RAIN_THRESHOLD_MM = 50.0

FORECAST_VARIABLES = [
    "temperature_2m_previous_day1",
    "relative_humidity_2m_previous_day1",
    "precipitation_previous_day1",
    "wind_speed_10m_previous_day1",
]

# --------------------------------------------------
# 2. Load observed rainfall
# --------------------------------------------------

print("Loading observed weather data...")

observed = pd.read_csv(OBSERVED_FILE)
observed["date"] = pd.to_datetime(observed["date"]).dt.strftime("%Y-%m-%d")

# Keep only the target and location information.
observed = observed[
    ["date", "location", "latitude", "longitude", "PRECTOTCORR"]
].copy()

observed = observed.rename(
    columns={"PRECTOTCORR": "actual_rainfall_mm"}
)

observed = observed[
    observed["date"].between(START_DATE, END_DATE)
]

print("Observed records in selected period:", len(observed))

# --------------------------------------------------
# 3. Download historical forecast data
# --------------------------------------------------

forecast_frames = []

locations = (
    observed[["location", "latitude", "longitude"]]
    .drop_duplicates(subset=["location"])
    .sort_values("location")
)

for _, place in locations.iterrows():
    location = place["location"]
    latitude = float(place["latitude"])
    longitude = float(place["longitude"])

    print(f"\nDownloading forecast data for {location}...")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": FORECAST_VARIABLES,
        "timezone": "Asia/Kathmandu",
    }

    try:
        response = requests.get(API_URL, params=params, timeout=180)

        if not response.ok:
            print(f"Request failed for {location}: HTTP {response.status_code}")
            print(response.text[:500])
            continue

        data = response.json()

        if "hourly" not in data:
            print(f"No hourly data returned for {location}.")
            print(data)
            continue

        hourly = pd.DataFrame(data["hourly"])

        if hourly.empty:
            print(f"Empty forecast data for {location}.")
            continue

        hourly["date"] = pd.to_datetime(hourly["time"]).dt.strftime(
            "%Y-%m-%d"
        )

        for column in FORECAST_VARIABLES:
            if column not in hourly.columns:
                hourly[column] = float("nan")

        # Require all 24 hours and valid values for all four variables.
        daily = hourly.groupby("date").agg(
            hourly_records=("time", "size"),
            temperature_mean_c=(
                "temperature_2m_previous_day1", "mean"
            ),
            humidity_mean_pct=(
                "relative_humidity_2m_previous_day1", "mean"
            ),
            forecast_rainfall_mm=(
                "precipitation_previous_day1", "sum"
            ),
            wind_speed_mean_kmh=(
                "wind_speed_10m_previous_day1", "mean"
            ),
            valid_temperature=(
                "temperature_2m_previous_day1", "count"
            ),
            valid_humidity=(
                "relative_humidity_2m_previous_day1", "count"
            ),
            valid_precipitation=(
                "precipitation_previous_day1", "count"
            ),
            valid_wind=(
                "wind_speed_10m_previous_day1", "count"
            ),
        ).reset_index()

        daily["location"] = location
        daily["latitude"] = latitude
        daily["longitude"] = longitude

        forecast_frames.append(daily)

        print("Days returned:", len(daily))

    except (requests.RequestException, ValueError, KeyError) as error:
        print(f"Error for {location}: {error}")

    # Avoid sending requests too quickly.
    time.sleep(1)

# --------------------------------------------------
# 4. Combine forecasts and observed rainfall
# --------------------------------------------------

if not forecast_frames:
    raise RuntimeError(
        "No forecast data was downloaded. Check the API response."
    )

forecast = pd.concat(forecast_frames, ignore_index=True)

forecast["complete_forecast"] = (
    forecast["hourly_records"].eq(24)
    & forecast["valid_temperature"].eq(24)
    & forecast["valid_humidity"].eq(24)
    & forecast["valid_precipitation"].eq(24)
    & forecast["valid_wind"].eq(24)
)

dataset = forecast.merge(
    observed,
    on=["date", "location", "latitude", "longitude"],
    how="inner",
    validate="one_to_one",
)

# Retain only complete forecasts and valid observed targets.
dataset = dataset[
    dataset["complete_forecast"]
    & dataset["actual_rainfall_mm"].notna()
].copy()

dataset["heavy_rain_tomorrow"] = (
    dataset["actual_rainfall_mm"] >= HEAVY_RAIN_THRESHOLD_MM
).astype(int)

# Keep only useful columns in the final dataset.
final_columns = [
    "date",
    "location",
    "latitude",
    "longitude",
    "temperature_mean_c",
    "humidity_mean_pct",
    "forecast_rainfall_mm",
    "wind_speed_mean_kmh",
    "actual_rainfall_mm",
    "heavy_rain_tomorrow",
]

dataset = dataset[final_columns].sort_values(
    ["location", "date"]
).reset_index(drop=True)

# --------------------------------------------------
# 5. Save and report
# --------------------------------------------------

dataset.to_csv(OUTPUT_FILE, index=False)

print("\n--- FORECAST DATASET SUMMARY ---")
print("Rows saved:", len(dataset))
print("Locations:", dataset["location"].nunique())
print("Date range:", dataset["date"].min(), "to", dataset["date"].max())
print("Heavy-rain cases:", dataset["heavy_rain_tomorrow"].sum())

print("\nRecords per location:")
print(dataset.groupby("location").size().to_string())

print("\nHeavy-rain cases per location:")
print(dataset.groupby("location")["heavy_rain_tomorrow"].sum().to_string())

print("\nSample records:")
print(dataset.head(10).to_string(index=False))

print("\nSaved to:", OUTPUT_FILE)
