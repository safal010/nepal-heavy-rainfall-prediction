
import requests
import pandas as pd
from pathlib import Path

# Project files
DATA_DIR = Path("data")
OBSERVED_FILE = DATA_DIR / "nepal_weather.csv"
OUTPUT_FILE = DATA_DIR / "forecast_comparison_previous_runs.csv"

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
MODEL = "ecmwf_ifs025"


def main():
    print("Loading observed rainfall data...")

    observed = pd.read_csv(OBSERVED_FILE)
    observed["date"] = pd.to_datetime(
        observed["date"]
    ).dt.date

    observed = observed[
        (observed["date"] >= pd.to_datetime(START_DATE).date())
        & (observed["date"] <= pd.to_datetime(END_DATE).date())
    ].copy()

    observed = observed.rename(
        columns={"PRECTOTCORR": "actual_rainfall_mm"}
    )

    locations = (
        observed.groupby("location")
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
        )
        .reset_index()
    )

    print("Locations:", len(locations))
    print("Model:", MODEL)
    print("Period:", START_DATE, "to", END_DATE)

    all_forecasts = []

    for _, location in locations.iterrows():
        name = location["location"]

        params = {
            "latitude": float(location["latitude"]),
            "longitude": float(location["longitude"]),
            "hourly": "precipitation_previous_day1",
            "start_date": START_DATE,
            "end_date": END_DATE,
            "timezone": "UTC",
            "models": MODEL,
        }

        print(f"\nDownloading forecasts for {name}...")

        response = requests.get(
            API_URL,
            params=params,
            timeout=120,
        )

        if not response.ok:
            raise RuntimeError(
                f"API failed for {name}: "
                f"{response.status_code} {response.text}"
            )

        data = response.json()
        hourly = data.get("hourly", {})

        times = hourly.get("time", [])
        rainfall = hourly.get(
            "precipitation_previous_day1", []
        )

        if len(times) != len(rainfall) or not times:
            raise ValueError(
                f"Unexpected hourly data for {name}"
            )

        hourly_df = pd.DataFrame({
            "time": pd.to_datetime(times),
            "forecast_rainfall_mm": rainfall,
        })

        hourly_df["date"] = hourly_df["time"].dt.date

        # Require 24 valid hourly values for each day.
        daily = (
            hourly_df.groupby("date")
            .agg(
                forecast_rainfall_mm=(
                    "forecast_rainfall_mm",
                    lambda values: values.sum(min_count=24),
                ),
                valid_hours=("forecast_rainfall_mm", "count"),
            )
            .reset_index()
        )

        daily.loc[
            daily["valid_hours"] != 24,
            "forecast_rainfall_mm",
        ] = float("nan")

        daily["location"] = name
        all_forecasts.append(daily)

        valid_days = daily["forecast_rainfall_mm"].notna().sum()

        print(
            f"{name}: {valid_days} complete forecast days "
            f"out of {len(daily)}"
        )

    forecasts = pd.concat(
        all_forecasts, ignore_index=True
    )

    # Keep the observed target and forecast date aligned.
    results = observed[
        ["date", "location", "actual_rainfall_mm"]
    ].merge(
        forecasts[
            [
                "date",
                "location",
                "forecast_rainfall_mm",
                "valid_hours",
            ]
        ],
        on=["date", "location"],
        how="left",
        validate="one_to_one",
    )

    results["absolute_error_mm"] = (
        results["actual_rainfall_mm"]
        - results["forecast_rainfall_mm"]
    ).abs()

    results.to_csv(OUTPUT_FILE, index=False)

    valid = results.dropna(
        subset=["actual_rainfall_mm", "forecast_rainfall_mm"]
    ).copy()

    print("\n--- EVALUATION SUMMARY ---")
    print("Expected comparisons:", len(results))
    print("Valid comparisons:", len(valid))
    print(
        "Missing forecast comparisons:",
        len(results) - len(valid),
    )

    if valid.empty:
        print("No valid comparisons; check the API response.")
        return

    forecast_mae = valid["absolute_error_mm"].mean()

    # Compare against zero rainfall on exactly the same dates.
    baseline_mae = valid["actual_rainfall_mm"].abs().mean()

    improvement = (
        (baseline_mae - forecast_mae) / baseline_mae * 100
        if baseline_mae != 0
        else float("nan")
    )

    print(f"Forecast MAE: {forecast_mae:.2f} mm")
    print(f"Zero-rainfall baseline MAE: {baseline_mae:.2f} mm")
    print(f"Improvement over baseline: {improvement:.2f}%")

    print("\n--- MAE BY LOCATION ---")
    print(
        valid.groupby("location")["absolute_error_mm"]
        .mean()
        .sort_values()
        .round(2)
    )

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
