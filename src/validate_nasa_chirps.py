
import os
import requests
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.io import MemoryFile

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/nepal_weather.csv"
OUTPUT_DIR = "data/chirps"

THRESHOLD_MM = 50.0
RANDOM_SEED = 42

REPRESENTATIVE_DAYS_PER_LOCATION = 30
HEAVY_DAYS_PER_LOCATION = 15

BASE_URL = (
    "https://data.chc.ucsb.edu/products/"
    "CHIRPS/v3.0/daily/final"
)

PRODUCTS = ["rnl", "sat"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD ORIGINAL DATASET (READ ONLY)
# ============================================================

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

required = [
    "date",
    "location",
    "latitude",
    "longitude",
    "PRECTOTCORR",
]

if df[required].isna().any().any():
    raise ValueError("Required dataset columns contain missing values.")

df = df.drop_duplicates(
    subset=["location", "date"]
).copy()

# ============================================================
# SELECT REPRESENTATIVE DATES
# ============================================================

representative_parts = []

for location, group in df.groupby("location"):
    count = min(
        REPRESENTATIVE_DAYS_PER_LOCATION,
        len(group),
    )

    representative_parts.append(
        group.sample(
            n=count,
            random_state=RANDOM_SEED,
        )
    )

representative = pd.concat(
    representative_parts,
    ignore_index=True,
)

# ============================================================
# SELECT HEAVY RAINFALL EVENTS
# ============================================================

heavy_parts = []

for location, group in df.groupby("location"):
    candidates = group[
        group["PRECTOTCORR"] >= THRESHOLD_MM
    ].sort_values(
        "PRECTOTCORR",
        ascending=False,
    )

    heavy_parts.append(
        candidates.head(HEAVY_DAYS_PER_LOCATION)
    )

heavy = pd.concat(
    heavy_parts,
    ignore_index=True,
)

# ============================================================
# DOWNLOAD AND READ CHIRPS V3 RASTER
# ============================================================

session = requests.Session()

# Cache complete raster bytes by product and date.
raster_cache = {}


def download_raster(product, date):
    """Download one CHIRPS v3 daily raster."""

    key = (product, date.strftime("%Y-%m-%d"))

    if key in raster_cache:
        return raster_cache[key]

    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")

    filename = (
        f"chirps-v3.0.{product}."
        f"{year}.{month}.{day}.tif"
    )

    url = (
        f"{BASE_URL}/{product}/{year}/{filename}"
    )

    response = session.get(
        url,
        timeout=180,
    )

    response.raise_for_status()

    if len(response.content) < 1000:
        raise ValueError("Downloaded raster appears too small.")

    raster_cache[key] = response.content

    return response.content


def extract_chirps_values(
    product,
    date,
    latitude,
    longitude,
):
    """Extract center pixel and valid 3x3 mean."""

    raster_bytes = download_raster(product, date)

    with MemoryFile(raster_bytes) as memfile:
        with memfile.open() as dataset:

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

            center = float(
                dataset.read(
                    1,
                    window=Window(col, row, 1, 1),
                )[0, 0]
            )

            row_start = max(0, row - 1)
            col_start = max(0, col - 1)

            row_end = min(
                dataset.height,
                row + 2,
            )

            col_end = min(
                dataset.width,
                col + 2,
            )

            nearby = dataset.read(
                1,
                window=Window(
                    col_start,
                    row_start,
                    col_end - col_start,
                    row_end - row_start,
                ),
            )

            # CHIRPS v3 missing-value sentinel is -9999.
            valid = (
                np.isfinite(nearby)
                & (nearby >= 0)
                & (nearby < 10000)
            )

            if dataset.nodata is not None:
                valid &= nearby != dataset.nodata

            if (
                not np.isfinite(center)
                or center < 0
                or center >= 10000
            ):
                raise ValueError(
                    f"Invalid center-pixel value: {center}"
                )

            if not valid.any():
                raise ValueError(
                    "No valid pixels in 3x3 neighborhood."
                )

            nearby_mean = float(
                nearby[valid].mean()
            )

    return center, nearby_mean


# ============================================================
# EVALUATE ONE SAMPLE FOR ONE PRODUCT
# ============================================================

def evaluate_sample(sample, sample_name, product):

    results = []

    print(
        f"\nEvaluating {sample_name} using "
        f"CHIRPS {product.upper()}"
    )

    print("Selected records:", len(sample))

    for _, event in sample.iterrows():

        date = event["date"]
        location = event["location"]

        nasa = float(event["PRECTOTCORR"])

        try:
            center, nearby_mean = extract_chirps_values(
                product,
                date,
                float(event["latitude"]),
                float(event["longitude"]),
            )

            results.append({
                "date": date.strftime("%Y-%m-%d"),
                "location": location,
                "product": product.upper(),
                "nasa_power_mm": nasa,
                "chirps_center_mm": center,
                "chirps_3x3_mean_mm": nearby_mean,
                "center_abs_error_mm": abs(
                    nasa - center
                ),
                "nearby_abs_error_mm": abs(
                    nasa - nearby_mean
                ),
                "nasa_heavy": nasa >= THRESHOLD_MM,
                "chirps_center_heavy": (
                    center >= THRESHOLD_MM
                ),
                "chirps_nearby_heavy": (
                    nearby_mean >= THRESHOLD_MM
                ),
            })

            print(
                f"{location} {date:%Y-%m-%d}: "
                f"NASA={nasa:.2f}, "
                f"CHIRPS={center:.2f}"
            )

        except Exception as error:

            print(
                f"Skipped {location} "
                f"{date:%Y-%m-%d}: {error}"
            )

    report = pd.DataFrame(results)

    if report.empty:
        print("No successful comparisons.")
        return report

    print("\n--- SUMMARY ---")

    print("Successful comparisons:", len(report))

    print(
        "Center-pixel MAE:",
        round(
            report["center_abs_error_mm"].mean(),
            2,
        ),
        "mm",
    )

    print(
        "Center-pixel median absolute error:",
        round(
            report["center_abs_error_mm"].median(),
            2,
        ),
        "mm",
    )

    print(
        "3x3 mean MAE:",
        round(
            report["nearby_abs_error_mm"].mean(),
            2,
        ),
        "mm",
    )

    print(
        "Center threshold disagreements:",
        int(
            (
                report["nasa_heavy"]
                != report["chirps_center_heavy"]
            ).sum()
        ),
    )

    return report


# ============================================================
# RUN VALIDATION FOR RNL AND SAT
# ============================================================

for product in PRODUCTS:

    print("\n" + "=" * 60)
    print("CHIRPS PRODUCT:", product.upper())
    print("=" * 60)

    representative_report = evaluate_sample(
        representative,
        "representative",
        product,
    )

    heavy_report = evaluate_sample(
        heavy,
        "heavy events",
        product,
    )

    representative_file = os.path.join(
        OUTPUT_DIR,
        f"validation_representative_{product}.csv",
    )

    heavy_file = os.path.join(
        OUTPUT_DIR,
        f"validation_heavy_events_{product}.csv",
    )

    if not representative_report.empty:
        if os.path.exists(representative_file):
            raise FileExistsError(
                f"Refusing to overwrite {representative_file}"
            )

        representative_report.to_csv(
            representative_file,
            index=False,
        )

        print("Saved:", representative_file)

    if not heavy_report.empty:
        if os.path.exists(heavy_file):
            raise FileExistsError(
                f"Refusing to overwrite {heavy_file}"
            )

        heavy_report.to_csv(
            heavy_file,
            index=False,
        )

        print("Saved:", heavy_file)

print("\nValidation finished.")
print("Original dataset and model files were not modified.")
