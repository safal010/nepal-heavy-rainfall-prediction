
import time
import requests
import pandas as pd
from pathlib import Path

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

INPUT_FILE = Path("data/nepal_weather.csv")
OUTPUT_FILE = Path("data/chirps/nasa_full_utc_lst_comparison.csv")

HEAVY_RAIN_THRESHOLD_MM = 50.0
LARGE_DIFFERENCE_MM = 5.0

def get_daily_rainfall(latitude, longitude, start_date, end_date, time_standard):
    params = {
        "parameters": "PRECTOTCORR",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": time_standard,
    }

    response = requests.get(URL, params=params, timeout=120)
    response.raise_for_status()

    values = response.json()["properties"]["parameter"]["PRECTOTCORR"]

    return pd.Series(
        {
            pd.to_datetime(date, format="%Y%m%d"): float(value)
            for date, value in values.items()
        },
        name=time_standard,
    )


def main():
    source = pd.read_csv(INPUT_FILE, parse_dates=["date"])

    locations = (
        source.groupby("location")
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
        )
        .reset_index()
    )

    start_date = source["date"].min()
    end_date = source["date"].max()

    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Locations: {len(locations)}")
    print("Downloading complete daily rainfall series...")

    reports = []

    for _, location_row in locations.iterrows():
        location = location_row["location"]
        latitude = float(location_row["latitude"])
        longitude = float(location_row["longitude"])

        print(f"\nProcessing {location}...")

        try:
            utc = get_daily_rainfall(
                latitude, longitude, start_date, end_date, "UTC"
            )
            time.sleep(1)

            lst = get_daily_rainfall(
                latitude, longitude, start_date, end_date, "LST"
            )
            time.sleep(1)

            comparison = pd.concat([utc, lst], axis=1).dropna()
            comparison = comparison.reset_index(names="date")

            comparison["location"] = location
            comparison["absolute_difference_mm"] = (
                comparison["UTC"] - comparison["LST"]
            ).abs()

            comparison["UTC_heavy"] = (
                comparison["UTC"] >= HEAVY_RAIN_THRESHOLD_MM
            )
            comparison["LST_heavy"] = (
                comparison["LST"] >= HEAVY_RAIN_THRESHOLD_MM
            )

            comparison["label_disagreement"] = (
                comparison["UTC_heavy"] != comparison["LST_heavy"]
            )

            comparison["difference_over_5mm"] = (
                comparison["absolute_difference_mm"] > LARGE_DIFFERENCE_MM
            )

            reports.append(comparison)
            print(f"Compared {len(comparison)} matching dates.")

        except Exception as error:
            print(f"ERROR for {location}: {error}")

    if not reports:
        print("\nNo results collected. Check the API response and try again.")
        return

    report = pd.concat(reports, ignore_index=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_FILE, index=False)

    print("\n--- FULL-DATE COMPARISON SUMMARY ---")
    print("Valid location-date comparisons:", len(report))
    print("Locations represented:", report["location"].nunique())

    print(
        "Mean absolute difference:",
        round(report["absolute_difference_mm"].mean(), 2),
        "mm",
    )
    print(
        "Median absolute difference:",
        round(report["absolute_difference_mm"].median(), 2),
        "mm",
    )

    print(
        "Days differing by more than 5 mm:",
        int(report["difference_over_5mm"].sum()),
    )
    print(
        "Percentage differing by more than 5 mm:",
        round(report["difference_over_5mm"].mean() * 100, 2),
        "%",
    )

    print(
        "Heavy-rain label disagreements:",
        int(report["label_disagreement"].sum()),
    )
    print(
        "Heavy-rain label disagreement rate:",
        round(report["label_disagreement"].mean() * 100, 3),
        "%",
    )

    print("\n--- DISAGREEMENTS BY LOCATION ---")
    summary = report.groupby("location").agg(
        compared_days=("date", "count"),
        large_differences=("difference_over_5mm", "sum"),
        label_disagreements=("label_disagreement", "sum"),
    )
    summary["label_disagreement_percent"] = (
        summary["label_disagreements"] / summary["compared_days"] * 100
    )
    print(summary.round(2).to_string())

    print(f"\nSaved report to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
