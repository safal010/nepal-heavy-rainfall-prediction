
import requests
import pandas as pd
from datetime import timedelta

locations = {
    "Kathmandu": (27.7172, 85.3240),
    "Pokhara": (28.2096, 83.9856),
    "Biratnagar": (26.4525, 87.2718),
    "Nepalgunj": (28.0500, 81.6167),
    "Janakpur": (26.7271, 85.9407),
    "Jumla": (29.2747, 82.1838),
}

url = "https://single-runs-api.open-meteo.com/v1/forecast"

observations = pd.read_csv("data/nepal_weather.csv")
observations["date"] = pd.to_datetime(
    observations["date"]
).dt.date

# Five target dates to test
target_dates = pd.date_range(
    "2025-09-02",
    "2025-09-06",
    freq="D",
)

results = []

for target_timestamp in target_dates:
    target_date = target_timestamp.date()

    # Initialize the forecast one day before the target date
    run_date = target_date - timedelta(days=1)
    run_time = run_date.strftime("%Y-%m-%d") + "T00:00"

    print(f"\nTarget date: {target_date}, forecast run: {run_time}")

    for city, (latitude, longitude) in locations.items():
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "precipitation",
            "timezone": "UTC",
            "models": "ecmwf_ifs",
            "run": run_time,
            "forecast_hours": 72,
        }

        try:
            response = requests.get(
                url,
                params=params,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            if "hourly" not in data:
                print(f"  No forecast returned for {city}")
                continue

            forecast = pd.DataFrame({
                "time": data["hourly"]["time"],
                "rainfall": data["hourly"]["precipitation"],
            })

            forecast["time"] = pd.to_datetime(
                forecast["time"], utc=True
            )

            # Select the target UTC calendar day
            day_start = pd.Timestamp(
                target_date, tz="UTC"
            )
            day_end = day_start + pd.Timedelta(days=1)

            day = forecast[
                (forecast["time"] >= day_start)
                & (forecast["time"] < day_end)
            ]

            if len(day) != 24 or day["rainfall"].isna().any():
                print(f"  Incomplete forecast for {city}")
                continue

            city_obs = observations[
                (observations["location"] == city)
                & (observations["date"] == target_date)
            ]

            if len(city_obs) != 1:
                print(f"  Missing or duplicate observation for {city}")
                continue

            observed = float(
                city_obs["PRECTOTCORR"].iloc[0]
            )

            forecast_total = float(day["rainfall"].sum())

            results.append({
                "date": target_date,
                "location": city,
                "forecast_mm": round(forecast_total, 2),
                "observed_mm": observed,
                "absolute_error_mm": abs(
                    forecast_total - observed
                ),
            })

        except requests.RequestException as error:
            print(f"  Request failed for {city}: {error}")

result_df = pd.DataFrame(results)

print("\n--- ALL COMPARISONS ---")

if result_df.empty:
    print("No valid comparisons were collected.")
else:
    print(result_df.to_string(index=False))

    print("\n--- OVERALL SUMMARY ---")
    print("Valid comparisons:", len(result_df))
    print(
        "Mean absolute error:",
        round(result_df["absolute_error_mm"].mean(), 2),
        "mm",
    )

    print("\n--- ERROR BY LOCATION ---")
    print(
        result_df.groupby("location")["absolute_error_mm"]
        .mean()
        .round(2)
        .sort_values()
        .to_string()
    )

    print("\n--- ERROR BY DATE ---")
    print(
        result_df.groupby("date")["absolute_error_mm"]
        .mean()
        .round(2)
        .to_string()
    )

    output_path = "data/forecast_comparison_sample.csv"
    result_df.to_csv(output_path, index=False)
    print("\nSaved comparison results to:", output_path)
