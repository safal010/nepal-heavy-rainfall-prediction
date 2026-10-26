
import json
import time
from pathlib import Path

import pandas as pd
import requests


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

OBSERVED_FILE = DATA_DIR / "nepal_weather.csv"
RESULT_FILE = DATA_DIR / "forecast_comparison_daily.csv"
CACHE_DIR = DATA_DIR / "forecast_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"

MODEL = "ecmwf_ifs"
YEAR = 2025

LOCATIONS = {
    "Kathmandu": (27.7172, 85.3240),
    "Pokhara": (28.2096, 83.9856),
    "Biratnagar": (26.4525, 87.2718),
    "Nepalgunj": (28.0500, 81.6167),
    "Janakpur": (26.7271, 85.9407),
    "Jumla": (29.2747, 82.1838),
}


# --------------------------------------------------
# 2. LOAD OBSERVED RAINFALL
# --------------------------------------------------

print("Loading observed weather data...")

observed = pd.read_csv(OBSERVED_FILE)

observed["date"] = pd.to_datetime(
    observed["date"]
).dt.strftime("%Y-%m-%d")

observed = observed[
    ["date", "location", "PRECTOTCORR"]
].copy()

observed.rename(
    columns={"PRECTOTCORR": "observed_rainfall_mm"},
    inplace=True,
)

observed["observed_rainfall_mm"] = pd.to_numeric(
    observed["observed_rainfall_mm"],
    errors="coerce",
)

observed = observed.dropna(
    subset=["observed_rainfall_mm"]
)

observed_lookup = {
    (row["date"], row["location"]):
        float(row["observed_rainfall_mm"])
    for _, row in observed.iterrows()
}

print(f"Loaded {len(observed)} observed records.")


# --------------------------------------------------
# 3. FETCH AND CACHE ONE FORECAST
# --------------------------------------------------

def get_forecast(location, latitude, longitude, target_date):
    """
    Fetch the previous day's 00:00 UTC ECMWF run.
    Calculate precipitation for the target UTC date.

    Returns:
        rainfall, "success"  if 24 valid hourly values exist
        None, "error"        if the API request fails
        None, "incomplete"   if hourly values are incomplete
    """

    run_date = (
        pd.Timestamp(target_date) - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    cache_file = CACHE_DIR / (
        f"{location}_{target_date}.json"
    )

    # Reuse successful cached forecasts.
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as file:
                cached = json.load(file)

            if cached.get("status") == "success":
                return (
                    cached["forecast_rainfall_mm"],
                    "success",
                )

        except (OSError, json.JSONDecodeError, KeyError):
            print(f"Invalid cache file; fetching again: {cache_file}")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "precipitation",
        "timezone": "UTC",
        "models": MODEL,
        "run": f"{run_date}T00:00",
        "forecast_hours": 72,
    }

    # Try at most twice for temporary request failures.
    for attempt in range(1, 3):
        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=60,
            )

            # Rate limiting: wait before retrying.
            if response.status_code == 429:
                print(
                    f"Rate limit for {location}, {target_date}. "
                    "Waiting 60 seconds..."
                )
                time.sleep(60)
                continue

            # Display the actual API error instead of only the URL.
            if response.status_code >= 400:
                print(
                    f"\nAPI ERROR | {location} | {target_date}"
                )
                print(f"Run requested: {run_date}T00:00 UTC")
                print(f"HTTP status: {response.status_code}")
                print(f"API message: {response.text[:1500]}")

                # HTTP 400 usually indicates a bad or unavailable
                # request, so repeating the identical request is
                # unlikely to help.
                if response.status_code == 400:
                    return None, "error"

                if attempt < 2:
                    time.sleep(3)
                    continue

                return None, "error"

            data = response.json()

            # Some API responses contain an error without an HTTP error.
            if "error" in data:
                print(
                    f"\nAPI MESSAGE | {location} | {target_date}"
                )
                print(data.get("reason", data["error"]))
                return None, "error"

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            rainfall = hourly.get("precipitation", [])

            daily_values = [
                float(value)
                for timestamp, value in zip(times, rainfall)
                if timestamp[:10] == target_date
                and value is not None
            ]

            # A daily total requires all 24 hourly observations.
            if len(daily_values) != 24:
                print(
                    f"\nINCOMPLETE FORECAST | {location} | "
                    f"{target_date}: received "
                    f"{len(daily_values)} of 24 hourly values."
                )
                return None, "incomplete"

            forecast_mm = sum(daily_values)

            # Cache only successful results.
            with open(
                cache_file, "w", encoding="utf-8"
            ) as file:
                json.dump(
                    {
                        "status": "success",
                        "forecast_rainfall_mm": forecast_mm,
                    },
                    file,
                )

            time.sleep(0.2)

            return forecast_mm, "success"

        except requests.RequestException as error:
            print(
                f"Request exception | {location} | "
                f"{target_date} | Attempt {attempt}/2: {error}"
            )

            if attempt < 2:
                time.sleep(3)
            else:
                return None, "error"

        except (ValueError, TypeError, KeyError) as error:
            print(
                f"Response parsing error | {location} | "
                f"{target_date}: {error}"
            )
            return None, "error"

    return None, "error"


# --------------------------------------------------
# 4. PREPARE DATES AND RESUME PROGRESS
# --------------------------------------------------

target_dates = pd.date_range(
    start=f"{YEAR}-01-01",
    end=f"{YEAR}-12-31",
    freq="D",
)

total = len(target_dates) * len(LOCATIONS)

completed_keys = set()
results = []

if RESULT_FILE.exists():
    previous = pd.read_csv(RESULT_FILE)

    required_columns = {"date", "location", "status"}

    if required_columns.issubset(previous.columns):
        # Keep only completed results. Failed/incomplete cases
        # will be retried.
        completed = previous[
            previous["status"].isin(["success", "missing"])
        ].copy()

        completed["date"] = pd.to_datetime(
            completed["date"]
        ).dt.strftime("%Y-%m-%d")

        completed = completed.drop_duplicates(
            subset=["date", "location"],
            keep="last",
        )

        # Exclude any unexpected dates or locations.
        valid_locations = set(LOCATIONS.keys())
        valid_dates = {
            date.strftime("%Y-%m-%d")
            for date in target_dates
        }

        completed = completed[
            completed["location"].isin(valid_locations)
            & completed["date"].isin(valid_dates)
        ]

        results = completed.to_dict("records")

        completed_keys = {
            (row["date"], row["location"])
            for row in results
        }

    print(f"Loaded {len(completed_keys)} completed records.")


# --------------------------------------------------
# 5. EVALUATE DAILY FORECASTS
# --------------------------------------------------

processed = len(completed_keys)

print(f"\nStarting evaluation for {YEAR}.")
print(f"Expected comparisons: {total}")
print("Successful forecasts are cached for future runs.")
print("Press Ctrl+C to stop safely.\n")

try:
    for target in target_dates:
        target_date = target.strftime("%Y-%m-%d")

        for location, coordinates in LOCATIONS.items():
            key = (target_date, location)

            if key in completed_keys:
                continue

            latitude, longitude = coordinates
            observed_mm = observed_lookup.get(key)

            if observed_mm is None:
                record = {
                    "date": target_date,
                    "location": location,
                    "forecast_rainfall_mm": None,
                    "observed_rainfall_mm": None,
                    "absolute_error_mm": None,
                    "status": "missing",
                }

            else:
                forecast_mm, status = get_forecast(
                    location,
                    latitude,
                    longitude,
                    target_date,
                )

                if status == "success":
                    record = {
                        "date": target_date,
                        "location": location,
                        "forecast_rainfall_mm": round(
                            forecast_mm, 3
                        ),
                        "observed_rainfall_mm": round(
                            observed_mm, 3
                        ),
                        "absolute_error_mm": round(
                            abs(forecast_mm - observed_mm), 3
                        ),
                        "status": "success",
                    }
                else:
                    record = {
                        "date": target_date,
                        "location": location,
                        "forecast_rainfall_mm": None,
                        "observed_rainfall_mm": round(
                            observed_mm, 3
                        ),
                        "absolute_error_mm": None,
                        "status": status,
                    }

            results.append(record)
            processed += 1

            # Save progress after each comparison.
            # Failed attempts remain in this CSV for inspection,
            # but are not treated as completed on the next run.
            pd.DataFrame(results).to_csv(
                RESULT_FILE,
                index=False,
            )

            if processed % 10 == 0 or processed == total:
                print(f"Progress: {processed}/{total}")

except KeyboardInterrupt:
    print("\nStopped by user. Progress has been saved.")


# --------------------------------------------------
# 6. SUMMARIZE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

if not results_df.empty:
    results_df = results_df.drop_duplicates(
        subset=["date", "location"],
        keep="last",
    )

    results_df.to_csv(RESULT_FILE, index=False)

    valid = results_df[
        results_df["status"] == "success"
    ].copy()

    print("\n--- EVALUATION SUMMARY ---")
    print(f"Records saved: {len(results_df)}")
    print(f"Valid comparisons: {len(valid)}")
    print(
        "Forecast errors/API failures: "
        f"{(results_df['status'] == 'error').sum()}"
    )
    print(
        "Incomplete forecasts: "
        f"{(results_df['status'] == 'incomplete').sum()}"
    )
    print(
        "Missing observations: "
        f"{(results_df['status'] == 'missing').sum()}"
    )

    if not valid.empty:
        forecast_mae = (
            valid["forecast_rainfall_mm"]
            - valid["observed_rainfall_mm"]
        ).abs().mean()

        baseline_mae = (
            valid["observed_rainfall_mm"].abs().mean()
        )

        print(f"\nForecast MAE: {forecast_mae:.2f} mm")
        print(f"Zero-rainfall baseline MAE: {baseline_mae:.2f} mm")

        if baseline_mae > 0:
            improvement = (
                (baseline_mae - forecast_mae)
                / baseline_mae
            ) * 100

            if improvement >= 0:
                print(
                    f"Forecast improves on baseline by "
                    f"{improvement:.1f}%."
                )
            else:
                print(
                    f"Forecast is worse than baseline by "
                    f"{abs(improvement):.1f}%."
                )

        print("\n--- MAE BY LOCATION ---")
        print(
            valid.groupby("location")["absolute_error_mm"]
            .mean()
            .sort_values()
            .round(2)
        )

    print(f"\nResults saved to: {RESULT_FILE}")

else:
    print("No results were collected.")
