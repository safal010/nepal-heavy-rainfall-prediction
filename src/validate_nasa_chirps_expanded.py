
import os
import time
import requests
import pandas as pd
import numpy as np

from rasterio.windows import Window
from rasterio.io import MemoryFile


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/nepal_weather.csv"

OUTPUT_DIR = "data/chirps/expanded"

SAMPLE_FILE = os.path.join(
    OUTPUT_DIR,
    "expanded_sample_dates.csv",
)

BASE_URL = (
    "https://data.chc.ucsb.edu/products/"
    "CHIRPS/v3.0/daily/final"
)

PRODUCTS = ["rnl", "sat"]

THRESHOLD_MM = 50.0

REQUEST_TIMEOUT = 180

EXPECTED_LOCATIONS = 6

# Retry configuration
MAX_DOWNLOAD_ATTEMPTS = 5

INITIAL_RETRY_DELAY = 15

MAX_RETRY_DELAY = 120

DELAY_BETWEEN_DATES = 2

RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# OUTPUT FILE PATHS
# ============================================================

def get_result_file(product):

    return os.path.join(
        OUTPUT_DIR,
        f"expanded_results_{product}.csv",
    )


def get_failure_file(product):

    return os.path.join(
        OUTPUT_DIR,
        f"expanded_failures_{product}.csv",
    )


# ============================================================
# LOAD EXISTING SAMPLE
# ============================================================

def load_sample():

    print("\nLoading existing expanded sample...")

    if not os.path.exists(SAMPLE_FILE):

        raise FileNotFoundError(
            f"Sample file not found: {SAMPLE_FILE}\n"
            "Please create the expanded sample first."
        )

    sample = pd.read_csv(SAMPLE_FILE)

    required_columns = [
        "date",
        "location",
        "latitude",
        "longitude",
        "PRECTOTCORR",
    ]

    missing = [
        column
        for column in required_columns
        if column not in sample.columns
    ]

    if missing:

        raise ValueError(
            f"Missing sample columns: {missing}"
        )

    sample["date"] = pd.to_datetime(
        sample["date"],
        errors="raise",
    )

    numeric_columns = [
        "latitude",
        "longitude",
        "PRECTOTCORR",
    ]

    for column in numeric_columns:

        sample[column] = pd.to_numeric(
            sample[column],
            errors="raise",
        )

    if sample[required_columns].isna().any().any():

        raise ValueError(
            "Sample contains missing required values."
        )

    if sample.duplicated(
        subset=["date", "location"]
    ).any():

        raise ValueError(
            "Duplicate date/location records found."
        )

    location_counts = (
        sample.groupby("date")["location"].nunique()
    )

    if not (location_counts == EXPECTED_LOCATIONS).all():

        raise ValueError(
            "Some dates do not contain all six locations."
        )

    print("Sample loaded successfully.")
    print("Total records:", len(sample))
    print("Unique dates:", sample["date"].nunique())
    print("Locations:", sample["location"].nunique())

    return sample.sort_values(
        ["date", "location"]
    ).reset_index(drop=True)


# ============================================================
# CALCULATE RETRY WAIT TIME
# ============================================================

def get_retry_wait_time(response, attempt):

    retry_after = response.headers.get("Retry-After")

    if retry_after:

        try:

            wait_time = float(retry_after)

            return min(
                max(wait_time, 1),
                MAX_RETRY_DELAY,
            )

        except ValueError:

            pass

    wait_time = INITIAL_RETRY_DELAY * (2 ** (attempt - 1))

    return min(
        wait_time,
        MAX_RETRY_DELAY,
    )


# ============================================================
# DOWNLOAD ONE DAILY CHIRPS RASTER
# ============================================================

def download_chirps_raster(session, product, date):

    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")

    filename = (
        f"chirps-v3.0.{product}."
        f"{year}.{month}.{day}.tif"
    )

    url = f"{BASE_URL}/{product}/{year}/{filename}"

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):

        print(
            f"Download attempt {attempt}/"
            f"{MAX_DOWNLOAD_ATTEMPTS}"
        )

        try:

            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT,
            )

            # Handle temporary server errors and rate limits
            if response.status_code in RETRYABLE_STATUS_CODES:

                status_code = response.status_code

                if attempt == MAX_DOWNLOAD_ATTEMPTS:

                    response.raise_for_status()

                wait_time = get_retry_wait_time(
                    response,
                    attempt,
                )

                response.close()

                print(
                    f"Server returned HTTP {status_code}."
                )

                print(
                    f"Waiting {wait_time:.0f} seconds before retry..."
                )

                time.sleep(wait_time)

                continue

            # Raise an error for other unsuccessful responses
            response.raise_for_status()

            raster_bytes = response.content

            response.close()

            if len(raster_bytes) < 1000:

                raise ValueError(
                    "Downloaded raster is unexpectedly small."
                )

            print("Download successful.")

            return raster_bytes

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as error:

            if attempt == MAX_DOWNLOAD_ATTEMPTS:

                raise

            wait_time = min(
                INITIAL_RETRY_DELAY * (2 ** (attempt - 1)),
                MAX_RETRY_DELAY,
            )

            print(
                "Temporary connection problem:",
                error,
            )

            print(
                f"Waiting {wait_time} seconds before retry..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        f"Download failed after {MAX_DOWNLOAD_ATTEMPTS} attempts."
    )


# ============================================================
# EXTRACT RAINFALL VALUES FROM A RASTER
# ============================================================

def extract_rainfall(dataset, latitude, longitude):

    row, col = dataset.index(
        longitude,
        latitude,
    )

    if not (
        0 <= row < dataset.height
        and 0 <= col < dataset.width
    ):

        raise ValueError(
            "Coordinates outside raster."
        )

    # Center pixel
    center = float(
        dataset.read(
            1,
            window=Window(col, row, 1, 1),
        )[0, 0]
    )

    # Boundary-safe 3x3 window
    row_start = max(0, row - 1)
    col_start = max(0, col - 1)

    row_end = min(dataset.height, row + 2)
    col_end = min(dataset.width, col + 2)

    nearby = dataset.read(
        1,
        window=Window(
            col_start,
            row_start,
            col_end - col_start,
            row_end - row_start,
        ),
    )

    # Valid pixels only
    valid = (
        np.isfinite(nearby)
        & (nearby >= 0)
        & (nearby < 10000)
        & (nearby != -9999)
    )

    if dataset.nodata is not None:

        valid &= nearby != dataset.nodata

    # Validate center pixel
    if (
        not np.isfinite(center)
        or center < 0
        or center >= 10000
        or center == -9999
    ):

        raise ValueError(
            f"Invalid center-pixel value: {center}"
        )

    if not valid.any():

        raise ValueError(
            "No valid pixels in nearby window."
        )

    nearby_mean = float(
        nearby[valid].mean()
    )

    return center, nearby_mean


# ============================================================
# PROCESS ONE DATE FOR ALL PENDING LOCATIONS
# ============================================================

def process_one_date(
    session,
    product,
    date,
    daily_rows,
):

    raster_bytes = download_chirps_raster(
        session,
        product,
        date,
    )

    results = []
    failures = []

    with MemoryFile(raster_bytes) as memfile:

        with memfile.open() as dataset:

            for _, row in daily_rows.iterrows():

                try:

                    center, nearby_mean = extract_rainfall(
                        dataset,
                        row["latitude"],
                        row["longitude"],
                    )

                    nasa_rainfall = float(
                        row["PRECTOTCORR"]
                    )

                    results.append({

                        "date": date.strftime("%Y-%m-%d"),

                        "location": row["location"],

                        "latitude": row["latitude"],

                        "longitude": row["longitude"],

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

                except Exception as error:

                    failures.append({

                        "date": date.strftime("%Y-%m-%d"),

                        "product": product,

                        "location": row["location"],

                        "error": str(error),

                    })

    return results, failures


# ============================================================
# SAVE RESULTS WITHOUT DUPLICATES
# ============================================================

def save_records(records, filename, key_columns):

    if not records:

        return

    new_data = pd.DataFrame(records)

    if os.path.exists(filename):

        old_data = pd.read_csv(filename)

        combined = pd.concat(
            [old_data, new_data],
            ignore_index=True,
        )

    else:

        combined = new_data

    combined = combined.drop_duplicates(
        subset=key_columns,
        keep="last",
    )

    combined.to_csv(
        filename,
        index=False,
    )


# ============================================================
# REMOVE FAILURES THAT HAVE NOW SUCCEEDED
# ============================================================

def clear_resolved_failures(
    failure_file,
    successful_records,
):

    if (
        not successful_records
        or not os.path.exists(failure_file)
    ):

        return

    failures = pd.read_csv(failure_file)

    successful_keys = {
        (
            record["date"],
            record["location"],
        )
        for record in successful_records
    }

    failure_keys = list(
        zip(
            failures["date"].astype(str),
            failures["location"].astype(str),
        )
    )

    keep = [
        key not in successful_keys
        for key in failure_keys
    ]

    failures = failures.loc[keep]

    failures.to_csv(
        failure_file,
        index=False,
    )


# ============================================================
# PROCESS ALL DATES FOR ONE PRODUCT
# ============================================================

def run_product_validation(product, sample):

    result_file = get_result_file(product)

    failure_file = get_failure_file(product)

    print("\n" + "=" * 60)
    print(f"STARTING CHIRPS {product.upper()} VALIDATION")
    print("=" * 60)

    # Load previously successful records
    completed = set()

    if os.path.exists(result_file):

        existing = pd.read_csv(result_file)

        if not {
            "date",
            "location",
        }.issubset(existing.columns):

            raise ValueError(
                f"Invalid result file: {result_file}"
            )

        existing["date"] = pd.to_datetime(
            existing["date"]
        ).dt.strftime("%Y-%m-%d")

        existing["location"] = (
            existing["location"].astype(str)
        )

        completed = {
            (
                str(date),
                str(location),
            )
            for date, location in zip(
                existing["date"],
                existing["location"],
            )
        }

    grouped = sample.groupby(
        "date",
        sort=True,
    )

    dates = list(grouped.groups.keys())

    total_dates = len(dates)

    print("Total dates:", total_dates)

    print(
        "Previously successful locations:",
        len(completed),
    )

    session = requests.Session()

    # Identify this script politely to the data server
    session.headers.update({
        "User-Agent": "Nepal-CHIRPS-Research-Validation/1.0"
    })

    try:

        for index, date in enumerate(dates, start=1):

            date = pd.Timestamp(date)

            date_string = date.strftime("%Y-%m-%d")

            daily_rows = grouped.get_group(date)

            # Only process locations not already successful
            completed_locations = {
                location
                for saved_date, location in completed
                if saved_date == date_string
            }

            pending_rows = daily_rows[
                ~daily_rows["location"].astype(str).isin(
                    completed_locations
                )
            ]

            if pending_rows.empty:

                continue

            print(
                f"\n[{index}/{total_dates}] "
                f"{product.upper()} | {date_string}"
            )

            print(
                "Locations remaining:",
                len(pending_rows),
            )

            try:

                results, failures = process_one_date(
                    session,
                    product,
                    date,
                    pending_rows,
                )

                # Save successful records immediately
                save_records(
                    results,
                    result_file,
                    ["date", "location"],
                )

                # Update completed records
                for record in results:

                    completed.add((
                        record["date"],
                        record["location"],
                    ))

                # Save current failures
                if failures:

                    save_records(
                        failures,
                        failure_file,
                        ["date", "location"],
                    )

                # Remove failures for locations that succeeded
                clear_resolved_failures(
                    failure_file,
                    results,
                )

                print(
                    "Successful locations:",
                    len(results),
                )

                print(
                    "Failed locations:",
                    len(failures),
                )

            except Exception as error:

                print(
                    "Date processing failed:",
                    error,
                )

                # Download or raster-level failure:
                # record every pending location
                failures = []

                for _, row in pending_rows.iterrows():

                    failures.append({

                        "date": date_string,

                        "product": product,

                        "location": row["location"],

                        "error": str(error),

                    })

                save_records(
                    failures,
                    failure_file,
                    ["date", "location"],
                )

            # Pause before processing another date
            time.sleep(DELAY_BETWEEN_DATES)

    finally:

        session.close()

    print("\nFinished CHIRPS", product.upper())

    print("Results saved to:", result_file)

    print("Failures saved to:", failure_file)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("\nCHIRPS EXPANDED VALIDATION")

    sample = load_sample()

    for product in PRODUCTS:

        run_product_validation(
            product,
            sample,
        )

    print("\n" + "=" * 60)
    print("ALL CHIRPS VALIDATION PROCESSING FINISHED")
    print("=" * 60)


if __name__ == "__main__":

    main()
