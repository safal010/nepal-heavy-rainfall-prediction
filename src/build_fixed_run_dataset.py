from datetime import date, timedelta
from pathlib import Path
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"

OBSERVED_FILE = Path("data/nepal_weather.csv")
OUTPUT_FILE = Path("data/fixed_run_forecast_training_data.csv")
ERROR_FILE = Path("data/fixed_run_forecast_errors.csv")

# ECMWF IFS archived runs are expected to be available from March 2024.
START_DATE = date(2024, 3, 15)
END_DATE = date(2025, 12, 31)

HEAVY_RAIN_THRESHOLD_MM = 50.0

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

# False processes the full date range. True processes a small sample only.
PILOT_MODE = False
PILOT_RECORDS = 6

SAVE_EVERY = 25
REQUEST_DELAY_SECONDS = 0.2


def create_session():
    """Reuse HTTP connections and retry temporary HTTP failures."""
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def get_daily_forecast(session, latitude, longitude, target_date):
    """
    Fetch target-date weather from the previous day's 00:00 UTC run.

    The API response must contain exactly 24 complete hourly records for
    the target date. This function does not verify the archive's actual
    publication/issue timestamp; it uses the requested run parameter.
    """
    run_date = target_date - timedelta(days=1)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": HOURLY_VARIABLES,
        "timezone": "Asia/Kathmandu",
        "models": "ecmwf_ifs",
        "run": f"{run_date.isoformat()}T00:00",
        "forecast_hours": 72,
    }

    response = session.get(API_URL, params=params, timeout=60)

    if response.status_code != 200:
        raise ValueError(
            f"API HTTP {response.status_code}: {response.text[:300]}"
        )

    data = response.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    day_indices = [
        i for i, timestamp in enumerate(times)
        if timestamp[:10] == target_date.isoformat()
    ]

    if len(day_indices) != 24:
        raise ValueError(
            f"Expected 24 target-day hours; found {len(day_indices)}"
        )

    values = {}

    for variable in HOURLY_VARIABLES:
        series = hourly.get(variable, [])

        if not series or max(day_indices) >= len(series):
            raise ValueError(f"Missing API series: {variable}")

        day_values = [series[i] for i in day_indices]

        if any(value is None for value in day_values):
            raise ValueError(
                f"{variable} contains missing target-day values"
            )

        values[variable] = day_values

    if any(value < 0 for value in values["precipitation"]):
        raise ValueError("Negative precipitation value encountered")

    return {
        "forecast_run_utc": f"{run_date.isoformat()}T00:00",
        "temperature_mean_c": sum(values["temperature_2m"]) / 24,
        "humidity_mean_pct": sum(values["relative_humidity_2m"]) / 24,
        "forecast_rainfall_mm": sum(values["precipitation"]),
        "wind_speed_mean_kmh": sum(values["wind_speed_10m"]) / 24,
    }


def save_progress(results, errors):
    """Save successful rows and errors, removing duplicate keys."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.drop_duplicates(
            subset=["date", "location"],
            keep="last",
        ).sort_values(["date", "location"])

    errors_df = pd.DataFrame(errors)
    if not errors_df.empty:
        errors_df = errors_df.drop_duplicates(
            subset=["date", "location"],
            keep="last",
        ).sort_values(["date", "location"])

    results_df.to_csv(OUTPUT_FILE, index=False)
    errors_df.to_csv(ERROR_FILE, index=False)


 

def load_existing_records():
    """Load prior results and errors, handling empty files safely."""

    # Load successful records.
    if OUTPUT_FILE.exists():
        try:
            existing = pd.read_csv(OUTPUT_FILE)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()

        if not existing.empty:
            required_columns = {"date", "location"}

            if not required_columns.issubset(existing.columns):
                raise ValueError(
                    f"{OUTPUT_FILE} must contain date and location columns."
                )

            existing["date"] = existing["date"].astype(str)
            existing["location"] = existing["location"].astype(str)

            existing = existing.drop_duplicates(
                subset=["date", "location"],
                keep="last",
            )

            results = existing.to_dict("records")
            completed = set(
                zip(existing["date"], existing["location"])
            )
        else:
            results = []
            completed = set()
    else:
        results = []
        completed = set()

    # Load errors; an empty error file is valid.
    if ERROR_FILE.exists():
        try:
            previous_errors = pd.read_csv(ERROR_FILE)
        except pd.errors.EmptyDataError:
            previous_errors = pd.DataFrame()

        if {"date", "location"}.issubset(previous_errors.columns):
            previous_errors["date"] = (
                previous_errors["date"].astype(str)
            )
            previous_errors["location"] = (
                previous_errors["location"].astype(str)
            )
            errors = previous_errors.to_dict("records")
        else:
            errors = []
    else:
        errors = []

    # Remove errors for records that have already succeeded.
    errors = [
        error for error in errors
        if (
            str(error.get("date")),
            str(error.get("location")),
        ) not in completed
    ]

    return results, completed, errors



def main():
    if not OBSERVED_FILE.exists():
        raise FileNotFoundError(
            f"Observed data file not found: {OBSERVED_FILE}"
        )

    observed = pd.read_csv(OBSERVED_FILE)

    required = {
        "date",
        "location",
        "latitude",
        "longitude",
        "PRECTOTCORR",
    }
    missing = required - set(observed.columns)
    if missing:
        raise ValueError(
            f"Observed dataset is missing columns: {sorted(missing)}"
        )

    observed["date"] = pd.to_datetime(
        observed["date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    observed = observed.dropna(
        subset=[
            "date",
            "location",
            "latitude",
            "longitude",
            "PRECTOTCORR",
        ]
    )

    observed = observed[
        observed["date"].between(
            START_DATE.isoformat(),
            END_DATE.isoformat(),
        )
    ].copy()

    if observed.duplicated(["date", "location"]).any():
        raise ValueError(
            "Observed data contains duplicate date/location rows"
        )

    observed = observed.sort_values(
        ["date", "location"]
    ).reset_index(drop=True)

    if PILOT_MODE:
        first_date = observed["date"].min()
        work = observed[observed["date"] == first_date].copy()

        if len(work) < PILOT_RECORDS:
            work = observed.head(PILOT_RECORDS).copy()

        print("PILOT MODE: processing a small sample only.")
    else:
        work = observed.copy()

    results, completed, errors = load_existing_records()

    # Keep only records that have not already succeeded.
    work = work[
        ~work.apply(
            lambda row: (
                str(row["date"]), str(row["location"])
            ) in completed,
            axis=1,
        )
    ].copy()

    total = len(work)
    print("Previously completed records:", len(completed))
    print("Records remaining:", total)
    print(f"Target period: {START_DATE} to {END_DATE}")
    print(f"Heavy-rain threshold: {HEAVY_RAIN_THRESHOLD_MM:.1f} mm")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Error log: {ERROR_FILE}")

    session = create_session()
    processed_since_save = 0

    try:
        for count, row in enumerate(work.to_dict("records"), start=1):
            target_date = date.fromisoformat(row["date"])
            key = (str(row["date"]), str(row["location"]))

            try:
                forecast = get_daily_forecast(
                    session,
                    float(row["latitude"]),
                    float(row["longitude"]),
                    target_date,
                )

                actual = float(row["PRECTOTCORR"])
                if actual < 0:
                    raise ValueError(
                        f"Negative observed rainfall: {actual}"
                    )

                results.append({
                    "date": row["date"],
                    "location": row["location"],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    **forecast,
                    "actual_rainfall_mm": actual,
                    "heavy_rain_tomorrow": int(
                        actual >= HEAVY_RAIN_THRESHOLD_MM
                    ),
                })

                completed.add(key)
                # Remove any old error for this record after success.
                errors = [
                    error for error in errors
                    if (
                        str(error.get("date")),
                        str(error.get("location")),
                    ) != key
                ]

                print(
                    f"[{count}/{total}] OK "
                    f"{row['location']} {row['date']}"
                )

            except (
                requests.RequestException,
                ValueError,
                KeyError,
            ) as error:
                errors.append({
                    "date": row["date"],
                    "location": row["location"],
                    "error": str(error),
                })

                print(
                    f"[{count}/{total}] FAILED "
                    f"{row['location']} {row['date']}: {error}"
                )

            processed_since_save += 1
            if processed_since_save >= SAVE_EVERY:
                save_progress(results, errors)
                processed_since_save = 0
                print(
                    f"Progress saved. Successful: {len(results)}; "
                    f"logged errors: {len(errors)}"
                )

            time.sleep(REQUEST_DELAY_SECONDS)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
    finally:
        save_progress(results, errors)
        session.close()

    final = pd.DataFrame(results)
    print("\n--- DATASET SUMMARY ---")
    print("Successful unique rows:", len(final))
    print("Failed records in error log:", len(errors))
    print("Output:", OUTPUT_FILE)
    print("Error log:", ERROR_FILE)

    if not final.empty:
        print("Date range:", final["date"].min(), "to", final["date"].max())
        print("Locations:", final["location"].nunique())
        print(
            "Heavy-rain cases:",
            int(final["heavy_rain_tomorrow"].sum()),
        )
        print(
            "Duplicate date/location rows:",
            int(final.duplicated(["date", "location"]).sum()),
        )


if __name__ == "__main__":
    main()
