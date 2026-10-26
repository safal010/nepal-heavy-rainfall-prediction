
import requests
import pandas as pd
from datetime import timedelta
from pathlib import Path

# --------------------------------------------------
# 1. LOCATIONS
# --------------------------------------------------

locations = {
    "Kathmandu": (27.7172, 85.3240),
    "Pokhara": (28.2096, 83.9856),
    "Biratnagar": (26.4525, 87.2718),
    "Nepalgunj": (28.0500, 81.6167),
    "Janakpur": (26.7271, 85.9407),
    "Jumla": (29.2747, 82.1838),
}

# --------------------------------------------------
# 2. API AND FILE SETTINGS
# --------------------------------------------------

url = "https://single-runs-api.open-meteo.com/v1/forecast"

observations_file = Path("data/nepal_weather.csv")
output_file = Path("data/forecast_comparison_year.csv")

# --------------------------------------------------
# 3. LOAD OBSERVED RAINFALL DATA
# --------------------------------------------------

print("Loading observed rainfall data...")

observations = pd.read_csv(observations_file)

observations["date"] = pd.to_datetime(
    observations["date"],
    errors="coerce",
).dt.date

# Check that the required columns exist.
required_columns = {
    "date",
    "location",
    "PRECTOTCORR",
}

missing_columns = required_columns - set(observations.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns in observations file: {missing_columns}"
    )

# --------------------------------------------------
# 4. CREATE 12 TARGET DATES
# --------------------------------------------------

# Evaluate the 15th day of every month in 2025.
target_dates = [
    pd.Timestamp(f"2025-{month:02d}-15")
    for month in range(1, 13)
]

print("\nDates to evaluate:")

for target in target_dates:
    print(target.date())

# --------------------------------------------------
# 5. COLLECT FORECAST COMPARISONS
# --------------------------------------------------

results = []

# Track why comparisons could not be collected.
missing_forecasts = 0
incomplete_forecasts = 0
missing_observations = 0
request_failures = 0

session = requests.Session()

for target in target_dates:

    target_date = target.date()

    # Use the forecast run from the previous day at 00:00 UTC.
    run_date = target_date - timedelta(days=1)
    run_time = run_date.strftime("%Y-%m-%d") + "T00:00"

    print(
        f"\nTarget: {target_date} | "
        f"Forecast run: {run_time}"
    )

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
            response = session.get(
                url,
                params=params,
                timeout=60,
            )

            response.raise_for_status()
            data = response.json()

        except requests.RequestException as error:
            print(f"  Request failed for {city}: {error}")
            request_failures += 1
            continue

        except ValueError as error:
            print(f"  Invalid API response for {city}: {error}")
            request_failures += 1
            continue

        # Check whether the API returned hourly data.
        if (
            "hourly" not in data
            or "time" not in data["hourly"]
            or "precipitation" not in data["hourly"]
        ):
            reason = data.get("reason", "Hourly data unavailable")
            print(f"  No forecast for {city}: {reason}")
            missing_forecasts += 1
            continue

        # Put hourly forecast data into a DataFrame.
        forecast = pd.DataFrame({
            "time": data["hourly"]["time"],
            "rainfall": data["hourly"]["precipitation"],
        })

        forecast["time"] = pd.to_datetime(
            forecast["time"],
            utc=True,
            errors="coerce",
        )

        forecast["rainfall"] = pd.to_numeric(
            forecast["rainfall"],
            errors="coerce",
        )

        # Select the target UTC calendar day.
        start = pd.Timestamp(target_date, tz="UTC")
        end = start + pd.Timedelta(days=1)

        day = forecast[
            (forecast["time"] >= start)
            & (forecast["time"] < end)
        ]

        # Require 24 hourly records and no missing rainfall values.
        if (
            len(day) != 24
            or day["rainfall"].isna().any()
        ):
            print(
                f"  Incomplete forecast for {city}: "
                f"{len(day)} hourly records"
            )
            incomplete_forecasts += 1
            continue

        # Find the observed rainfall for the same location and date.
        actual_rows = observations[
            (observations["location"] == city)
            & (observations["date"] == target_date)
        ]

        if len(actual_rows) != 1:
            print(
                f"  Missing or duplicate observation for {city}: "
                f"{len(actual_rows)} rows"
            )
            missing_observations += 1
            continue

        observed_value = actual_rows["PRECTOTCORR"].iloc[0]

        if pd.isna(observed_value):
            print(f"  Observed rainfall is missing for {city}")
            missing_observations += 1
            continue

        # Sum the 24 hourly forecast values to obtain daily rainfall.
        predicted = float(day["rainfall"].sum())
        observed = float(observed_value)

        # Absolute error = distance between prediction and observation.
        absolute_error = abs(predicted - observed)

        results.append({
            "date": str(target_date),
            "location": city,
            "forecast_run": run_time,
            "forecast_mm": predicted,
            "observed_mm": observed,
            "absolute_error_mm": absolute_error,
        })

        print(
            f"  {city}: forecast={predicted:.2f} mm, "
            f"observed={observed:.2f} mm, "
            f"error={absolute_error:.2f} mm"
        )

# --------------------------------------------------
# 6. SAVE AND EVALUATE RESULTS
# --------------------------------------------------

session.close()

print("\n" + "=" * 45)
print("DATA COLLECTION SUMMARY")
print("=" * 45)

print("Expected comparisons:", len(target_dates) * len(locations))
print("Valid comparisons:", len(results))
print("Unavailable forecasts:", missing_forecasts)
print("Incomplete forecasts:", incomplete_forecasts)
print("Missing/duplicate observations:", missing_observations)
print("Request/API failures:", request_failures)

if not results:
    print("\nNo valid comparisons collected.")
    print("Check the error messages above.")
else:
    df = pd.DataFrame(results)

    # Save the results without overwriting the original dataset.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    # --------------------------------------------------
    # 7. OVERALL ERROR
    # --------------------------------------------------

    forecast_mae = df["absolute_error_mm"].mean()

    # Baseline: predict 0 mm of rainfall for every comparison.
    baseline_mae = df["observed_mm"].abs().mean()

    print("\n" + "=" * 45)
    print("YEAR-ROUND SUMMARY")
    print("=" * 45)

    print("Valid comparisons:", len(df))
    print(f"Forecast MAE: {forecast_mae:.2f} mm")
    print(f"Zero-rainfall baseline MAE: {baseline_mae:.2f} mm")

    if forecast_mae < baseline_mae:
        improvement = (
            (baseline_mae - forecast_mae) / baseline_mae * 100
            if baseline_mae > 0
            else 0
        )

        print(
            f"Result: Forecast beats the baseline by "
            f"{improvement:.1f}% on this sample."
        )

    elif forecast_mae > baseline_mae:
        print("Result: Forecast does not beat the baseline on this sample.")

    else:
        print("Result: Forecast and baseline have equal MAE.")

    # --------------------------------------------------
    # 8. ERROR BY MONTH
    # --------------------------------------------------

    df["month"] = pd.to_datetime(df["date"]).dt.month

    print("\n--- MAE BY MONTH ---")

    monthly_mae = (
        df.groupby("month")["absolute_error_mm"]
        .agg(["mean", "count"])
        .rename(columns={
            "mean": "MAE_mm",
            "count": "comparisons",
        })
        .round(2)
    )

    print(monthly_mae.to_string())

    # --------------------------------------------------
    # 9. ERROR BY LOCATION
    # --------------------------------------------------

    print("\n--- MAE BY LOCATION ---")

    location_mae = (
        df.groupby("location")["absolute_error_mm"]
        .agg(["mean", "count"])
        .rename(columns={
            "mean": "MAE_mm",
            "count": "comparisons",
        })
        .sort_values("MAE_mm")
        .round(2)
    )

    print(location_mae.to_string())

    # --------------------------------------------------
    # 10. FINAL FILE LOCATION
    # --------------------------------------------------

    print(f"\nSaved comparison results to: {output_file}")
