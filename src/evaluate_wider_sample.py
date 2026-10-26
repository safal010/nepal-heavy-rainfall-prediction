
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

# Ten dates spread across September
target_dates = pd.date_range(
    "2025-09-02", "2025-09-29", periods=10
)

results = []

for target in target_dates:
    target_date = target.date()
    run_date = target_date - timedelta(days=1)
    run_time = run_date.strftime("%Y-%m-%d") + "T00:00"

    print(f"\nTarget: {target_date} | Run: {run_time}")

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
                url, params=params, timeout=60
            )
            response.raise_for_status()
            data = response.json()

            if "hourly" not in data:
                print(f"  No forecast: {city}")
                continue

            forecast = pd.DataFrame({
                "time": data["hourly"]["time"],
                "rainfall": data["hourly"]["precipitation"],
            })

            forecast["time"] = pd.to_datetime(
                forecast["time"], utc=True
            )

            start = pd.Timestamp(target_date, tz="UTC")
            end = start + pd.Timedelta(days=1)

            day = forecast[
                (forecast["time"] >= start)
                & (forecast["time"] < end)
            ]

            if len(day) != 24 or day["rainfall"].isna().any():
                print(f"  Incomplete forecast: {city}")
                continue

            actual_rows = observations[
                (observations["location"] == city)
                & (observations["date"] == target_date)
            ]

            if len(actual_rows) != 1:
                print(f"  Missing/duplicate observation: {city}")
                continue

            actual = float(
                actual_rows["PRECTOTCORR"].iloc[0]
            )
            predicted = float(day["rainfall"].sum())

            results.append({
                "date": str(target_date),
                "location": city,
                "forecast_mm": predicted,
                "observed_mm": actual,
            })

        except (requests.RequestException, ValueError) as error:
            print(f"  Error for {city}: {error}")

df = pd.DataFrame(results)

if df.empty:
    print("\nNo valid comparisons collected.")
else:
    df["absolute_error_mm"] = (
        df["forecast_mm"] - df["observed_mm"]
    ).abs()

    print("\n--- SUMMARY ---")
    print("Valid comparisons:", len(df))
    print(
        "Forecast MAE:",
        round(df["absolute_error_mm"].mean(), 2), "mm"
    )
    print(
        "Zero-rainfall baseline MAE:",
        round(df["observed_mm"].abs().mean(), 2), "mm"
    )

    print("\n--- MAE BY LOCATION ---")
    print(
        df.groupby("location")["absolute_error_mm"]
        .mean().round(2).sort_values().to_string()
    )

    df.to_csv(
        "data/forecast_comparison_wider.csv",
        index=False,
    )
    print("\nSaved to data/forecast_comparison_wider.csv")
