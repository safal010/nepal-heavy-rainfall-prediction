
import requests
import pandas as pd

# Six locations and their coordinates
locations = {
    "Kathmandu": (27.7172, 85.3240),
    "Pokhara": (28.2096, 83.9856),
    "Biratnagar": (26.4525, 87.2718),
    "Nepalgunj": (28.0500, 81.6167),
    "Janakpur": (26.7271, 85.9407),
    "Jumla": (29.2747, 82.1838),
}

# Use the same historical forecast run for every location
url = "https://single-runs-api.open-meteo.com/v1/forecast"

params = {
    "hourly": "precipitation",
    "timezone": "UTC",
    "models": "ecmwf_ifs",
    "run": "2025-09-01T00:00",
    "forecast_hours": 96,
}

# Load the existing NASA POWER observations
observations = pd.read_csv("data/nepal_weather.csv")
observations["date"] = pd.to_datetime(
    observations["date"]
).dt.date

# Compare two complete forecast days
target_dates = [
    pd.Timestamp("2025-09-02").date(),
    pd.Timestamp("2025-09-03").date(),
]

results = []

for city, (latitude, longitude) in locations.items():
    print(f"Checking {city}...")

    request_params = params.copy()
    request_params["latitude"] = latitude
    request_params["longitude"] = longitude

    response = requests.get(
        url,
        params=request_params,
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            f"No hourly forecast returned for {city}: {data}"
        )

    # Convert hourly forecast into a DataFrame
    forecast = pd.DataFrame({
        "time": data["hourly"]["time"],
        "forecast_rainfall_mm": data["hourly"]["precipitation"],
    })

    forecast["time"] = pd.to_datetime(
        forecast["time"], utc=True
    )
    forecast["date"] = forecast["time"].dt.date

    # Calculate daily rainfall totals
    daily_forecast = forecast.groupby("date").agg(
        hourly_records=("time", "count"),
        valid_hours=("forecast_rainfall_mm", "count"),
        forecast_rainfall_mm=(
            "forecast_rainfall_mm",
            lambda values: values.sum(min_count=1),
        ),
    )

    # Select observed rainfall for this location
    city_observations = observations[
        observations["location"] == city
    ].set_index("date")

    for date in target_dates:
        if date not in daily_forecast.index:
            print(f"  No forecast for {date}")
            continue

        day = daily_forecast.loc[date]

        # Only compare complete days with 24 valid hours
        if day["hourly_records"] != 24 or day["valid_hours"] != 24:
            print(f"  Skipping incomplete forecast: {date}")
            continue

        if date not in city_observations.index:
            actual = float("nan")
        else:
            actual = city_observations.loc[
                date, "PRECTOTCORR"
            ]

        results.append({
            "location": city,
            "date": date,
            "forecast_mm": round(
                day["forecast_rainfall_mm"], 2
            ),
            "observed_mm": actual,
        })

# Display results
result_df = pd.DataFrame(results)

print("\n--- FORECAST VS OBSERVATION ---")

if result_df.empty:
    print("No complete matching forecast days were found.")
else:
    print(
        result_df.to_string(index=False)
    )

    print("\n--- SUMMARY ---")
    print("Number of comparisons:", len(result_df))

    valid = result_df.dropna(
        subset=["forecast_mm", "observed_mm"]
    )

    if not valid.empty:
        valid = valid.copy()
        valid["absolute_error_mm"] = (
            valid["forecast_mm"] - valid["observed_mm"]
        ).abs()

        print(
            "Mean absolute difference:",
            round(valid["absolute_error_mm"].mean(), 2),
            "mm",
        )

        print("\n--- RESULTS BY LOCATION ---")
        print(
            valid.groupby("location")[
                "absolute_error_mm"
            ].mean().round(2).to_string()
        )
