
import sys
from pathlib import Path

import pandas as pd
import requests
from rasterio.io import MemoryFile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_nasa_chirps_expanded import (
    download_chirps_raster,
    extract_rainfall,
    THRESHOLD_MM,
)

PRODUCT = "rnl"
DATE = "2025-04-01"

FOLDER = Path("data/chirps/expanded")

SAMPLE_FILE = FOLDER / "expanded_sample_dates.csv"
RESULT_FILE = FOLDER / "expanded_results_rnl.csv"
FAILURE_FILE = FOLDER / "expanded_failures_rnl.csv"


def main():

    print("RETRYING RNL:", DATE)

    # Load the original sample
    sample = pd.read_csv(SAMPLE_FILE)

    sample["date"] = pd.to_datetime(
        sample["date"]
    ).dt.strftime("%Y-%m-%d")

    daily_rows = sample[
        sample["date"] == DATE
    ].copy()

    if len(daily_rows) != 6:
        raise ValueError(
            f"Expected 6 locations, found {len(daily_rows)}"
        )

    # Check which locations are already successful
    existing = pd.read_csv(RESULT_FILE)

    existing["date"] = pd.to_datetime(
        existing["date"]
    ).dt.strftime("%Y-%m-%d")

    completed = set(
        zip(
            existing["date"],
            existing["location"].astype(str),
        )
    )

    pending_rows = daily_rows[
        ~daily_rows.apply(
            lambda row: (
                row["date"],
                str(row["location"]),
            ) in completed,
            axis=1,
        )
    ]

    if pending_rows.empty:
        print("All six locations are already completed.")
        return

    print("Locations remaining:", len(pending_rows))

    session = requests.Session()

    session.headers.update({
        "User-Agent": "Nepal-CHIRPS-Research-Validation/1.0"
    })

    try:

        # Download the raster
        raster_bytes = download_chirps_raster(
            session,
            PRODUCT,
            pd.to_datetime(DATE).date(),
        )

        results = []

        # Open downloaded TIFF using Rasterio
        with MemoryFile(raster_bytes) as memfile:

            with memfile.open() as dataset:

                for _, row in pending_rows.iterrows():

                    latitude = float(row["latitude"])
                    longitude = float(row["longitude"])

                    center, nearby_mean = extract_rainfall(
                        dataset,
                        latitude,
                        longitude,
                    )

                    nasa_rainfall = float(
                        row["PRECTOTCORR"]
                    )

                    results.append({

                        "date": DATE,

                        "location": row["location"],

                        "latitude": latitude,

                        "longitude": longitude,

                        "nasa_rainfall_mm": nasa_rainfall,

                        "chirps_center_mm": center,

                        "chirps_3x3_mean_mm": nearby_mean,

                        "heavy_nasa": int(
                            nasa_rainfall >= THRESHOLD_MM
                        ),

                        "heavy_chirps_center": int(
                            center >= THRESHOLD_MM
                        ),

                        "heavy_chirps_3x3": int(
                            nearby_mean >= THRESHOLD_MM
                        ),

                    })

        # Save without overwriting existing successful records
        new_data = pd.DataFrame(results)

        combined = pd.concat(
            [existing, new_data],
            ignore_index=True,
        )

        combined = combined.drop_duplicates(
            subset=["date", "location"],
            keep="last",
        )

        combined.to_csv(
            RESULT_FILE,
            index=False,
        )

        # Remove only failures that have now succeeded
        if FAILURE_FILE.exists():

            failures = pd.read_csv(FAILURE_FILE)

            successful_keys = {
                (DATE, str(record["location"]))
                for record in results
            }

            failure_keys = list(
                zip(
                    pd.to_datetime(
                        failures["date"]
                    ).dt.strftime("%Y-%m-%d"),
                    failures["location"].astype(str),
                )
            )

            keep = [
                key not in successful_keys
                for key in failure_keys
            ]

            failures = failures.loc[keep]

            failures.to_csv(
                FAILURE_FILE,
                index=False,
            )

        print("\nRetry successful!")
        print("New records saved:", len(results))
        print("Results file:", RESULT_FILE)

    finally:
        session.close()


if __name__ == "__main__":
    main()
