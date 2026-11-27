
import requests
import pandas as pd
import time
from pathlib import Path

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

INPUT_FILE = Path("data/nepal_weather.csv")
OUTPUT_FILE = Path("data/chirps/nasa_utc_lst_comparison.csv")

HEAVY_RAIN_THRESHOLD_MM = 50.0

def get_rainfall(latitude, longitude, start_date, end_date, time_standard):
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

    response = requests.get(URL, params=params, timeout=90)
    response.raise_for_status()

    data = response.json()["properties"]["parameter"]["PRECTOTCORR"]

    return {
        pd.to_datetime(date, format="%Y%m%d"): value
        for date, value in data.items()
    }


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

    # Use a balanced sample of dates from the existing dataset.
    # This keeps the first API comparison manageable.
    sample_parts = []

    for location, group in df.groupby("location"):
        group = group.sort_values("date")

        if len(group) > 0:
            sample_parts.append(
                group.iloc[::max(1, len(group) // 60)].head(60)
            )

    sample = pd.concat(sample_parts, ignore_index=True)
    sample = sample.drop_duplicates(["location", "date"])

    print("Sample rows:", len(sample))
    print("Locations:", sample["location"].nunique())
    print("\nDownloading UTC and LST values...")

    results = []

    for location, group in sample.groupby("location"):
        latitude = float(group["latitude"].iloc[0])
        longitude = float(group["longitude"].iloc[0])

        start_date = group["date"].min()
        end_date = group["date"].max()

        print(f"\n{location}: {start_date.date()} to {end_date.date()}")

        try:
            utc_values = get_rainfall(
                latitude, longitude, start_date, end_date, "UTC"
            )
            time.sleep(1)

            lst_values = get_rainfall(
                latitude, longitude, start_date, end_date, "LST"
            )
            time.sleep(1)

            for _, row in group.iterrows():
                date = row["date"]

                utc_rain = utc_values.get(date)
                lst_rain = lst_values.get(date)

                if utc_rain is None or lst_rain is None:
                    continue

                results.append({
                    "date": date.date(),
                    "location": location,
                    "NASA_UTC_mm": float(utc_rain),
                    "NASA_LST_mm": float(lst_rain),
                    "UTC_heavy": float(utc_rain) >= HEAVY_RAIN_THRESHOLD_MM,
                    "LST_heavy": float(lst_rain) >= HEAVY_RAIN_THRESHOLD_MM,
                    "absolute_difference_mm": abs(
                        float(utc_rain) - float(lst_rain)
                    ),
                    "label_disagreement": (
                        (float(utc_rain) >= HEAVY_RAIN_THRESHOLD_MM)
                        != (float(lst_rain) >= HEAVY_RAIN_THRESHOLD_MM)
                    ),
                })

            print("Completed.")

        except Exception as error:
            print(f"Could not retrieve {location}: {error}")

    if not results:
        print("\nNo comparison results were collected.")
        return

    report = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_FILE, index=False)

    print("\n--- COMPARISON SUMMARY ---")
    print("Valid comparisons:", len(report))
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
        "Heavy-rain label disagreements:",
        int(report["label_disagreement"].sum()),
    )
    print(
        "Label disagreement rate:",
        f"{report['label_disagreement'].mean() * 100:.2f}%",
    )

    print("\n--- DISAGREEMENTS BY LOCATION ---")
    print(
        report.groupby("location")["label_disagreement"]
        .agg(["sum", "count", "mean"])
        .assign(disagreement_percent=lambda x: x["mean"] * 100)
        .round(2)
    )

    print(f"\nSaved comparison to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
