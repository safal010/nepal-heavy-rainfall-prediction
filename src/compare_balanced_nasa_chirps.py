
import os
import gzip
import requests
import pandas as pd
import rasterio
from rasterio.windows import Window

DATA_FILE = "data/nepal_weather.csv"
OUTPUT_FILE = "data/chirps/balanced_nasa_chirps_comparison.csv"
THRESHOLD_MM = 50.0

# Load the original dataset
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

# Group rainfall into three categories
df["category"] = pd.cut(
    df["PRECTOTCORR"],
    bins=[-0.001, 1, 50, float("inf")],
    labels=["Dry/light", "Moderate", "Heavy"],
)

# Select up to two dates per location and category
sample = pd.concat(
    [
        group.sample(n=min(2, len(group)), random_state=42)
        for _, group in df.groupby(
            ["location", "category"], observed=True
        )
    ],
    ignore_index=True,
)

os.makedirs("data/chirps", exist_ok=True)

base_url = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    "global_daily/tifs/p05"
)

results = []

print("Selected records:", len(sample))
print("Starting NASA POWER vs CHIRPS comparison...")

for _, event in sample.iterrows():
    location = event["location"]
    date = event["date"]
    nasa_rain = float(event["PRECTOTCORR"])
    latitude = float(event["latitude"])
    longitude = float(event["longitude"])

    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    date_text = date.strftime("%Y-%m-%d")

    url = (
        f"{base_url}/{year}/"
        f"chirps-v2.0.{year}.{month}.{day}.tif.gz"
    )

    gz_file = f"data/chirps/temp_{year}{month}{day}.tif.gz"
    tif_file = f"data/chirps/temp_{year}{month}{day}.tif"

    print(
        f"\n{location} | {date_text} | "
        f"NASA POWER: {nasa_rain:.2f} mm"
    )

    try:
        # Download the daily CHIRPS raster
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        with open(gz_file, "wb") as file:
            file.write(response.content)

        # Decompress the downloaded raster
        with gzip.open(gz_file, "rb") as source:
            with open(tif_file, "wb") as destination:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)

        # Read the raster and calculate pixel statistics
        with rasterio.open(tif_file) as dataset:
            row, col = dataset.index(longitude, latitude)

            # Check that the location is inside the raster
            if not (
                0 <= row < dataset.height
                and 0 <= col < dataset.width
            ):
                print("Location outside raster; skipping.")
                continue

            # Read rainfall at the exact location
            value = float(
                dataset.read(
                    1,
                    window=Window(col, row, 1, 1),
                )[0, 0]
            )

            # Check the center pixel value
            if not 0 <= value < 10000:
                print("Invalid CHIRPS center value; skipping.")
                continue

            # Define a window around the center pixel
            row_start = max(0, row - 1)
            col_start = max(0, col - 1)

            height = min(3, dataset.height - row_start)
            width = min(3, dataset.width - col_start)

            # Read the surrounding 3x3 pixels
            nearby = dataset.read(
                1,
                window=Window(
                    col_start,
                    row_start,
                    width,
                    height,
                ),
            )

            # Exclude negative values and invalid large values
            valid_nearby = nearby[
                (nearby >= 0) & (nearby < 10000)
            ]

            if valid_nearby.size == 0:
                print("No valid nearby CHIRPS pixels; skipping.")
                continue

            # Calculate statistics for valid nearby pixels
            nearby_mean = float(valid_nearby.mean())
            nearby_min = float(valid_nearby.min())
            nearby_max = float(valid_nearby.max())

        # Compare NASA POWER with the CHIRPS center pixel
        difference = abs(nasa_rain - value)

        # Save both center and nearby-pixel results
        results.append(
            {
                "date": date_text,
                "location": location,
                "category": event["category"],
                "nasa_power_mm": nasa_rain,
                "chirps_mm": value,
                "absolute_difference_mm": difference,
                "nasa_heavy": nasa_rain >= THRESHOLD_MM,
                "chirps_heavy": value >= THRESHOLD_MM,
                "chirps_nearby_mean_mm": nearby_mean,
                "chirps_nearby_min_mm": nearby_min,
                "chirps_nearby_max_mm": nearby_max,
                "nearby_mean_absolute_difference_mm": abs(
                    nasa_rain - nearby_mean
                ),
                "nearby_heavy": nearby_mean >= THRESHOLD_MM,
            }
        )

        print(
            f"CHIRPS center: {value:.2f} mm | "
            f"3x3 mean: {nearby_mean:.2f} mm | "
            f"Center difference: {difference:.2f} mm"
        )

    except Exception as error:
        print("Could not check this record:", error)

    finally:
        # Remove temporary downloads after each record
        for path in (gz_file, tif_file):
            if os.path.exists(path):
                os.remove(path)

# Save results and summarize the comparison
if results:
    report = pd.DataFrame(results)
    report.to_csv(OUTPUT_FILE, index=False)

    print("\n=== CENTER PIXEL SUMMARY ===")
    print("Successful comparisons:", len(report))

    print("\nResults by rainfall category:")
    print(
        report.groupby("category", observed=True)[
            "absolute_difference_mm"
        ]
        .agg(["count", "mean", "median", "max"])
        .round(2)
    )

    print(
        "\nCenter pixel mean absolute difference:",
        round(report["absolute_difference_mm"].mean(), 2),
        "mm",
    )

    print(
        "Center pixel median absolute difference:",
        round(report["absolute_difference_mm"].median(), 2),
        "mm",
    )

    center_disagreements = (
        report["nasa_heavy"] != report["chirps_heavy"]
    ).sum()

    print(
        "Center pixel heavy-rain disagreements:",
        int(center_disagreements),
    )

    print("\n=== SURROUNDING 3x3 PIXEL SUMMARY ===")

    print(
        "Nearby mean absolute difference:",
        round(
            report[
                "nearby_mean_absolute_difference_mm"
            ].mean(),
            2,
        ),
        "mm",
    )

    print(
        "Nearby mean median absolute difference:",
        round(
            report[
                "nearby_mean_absolute_difference_mm"
            ].median(),
            2,
        ),
        "mm",
    )

    nearby_disagreements = (
        report["nasa_heavy"] != report["nearby_heavy"]
    ).sum()

    print(
        "Nearby mean heavy-rain disagreements:",
        int(nearby_disagreements),
    )

    print("\nReport saved to:", OUTPUT_FILE)

else:
    print("\nNo valid comparisons were completed.")
