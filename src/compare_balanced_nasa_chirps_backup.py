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

    print(f"\n{location} | {date_text} | NASA: {nasa_rain:.2f} mm")

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        with open(gz_file, "wb") as file:
            file.write(response.content)

        with gzip.open(gz_file, "rb") as source:
            with open(tif_file, "wb") as destination:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)

        with rasterio.open(tif_file) as dataset:
            row, col = dataset.index(longitude, latitude)

            # Check that the grid cell is inside the raster
            if not (
                0 <= row < dataset.height
                and 0 <= col < dataset.width
            ):
                print("Location outside raster; skipping.")
                continue

            value = float(
                dataset.read(
                    1, window=Window(col, row, 1, 1)
                )[0, 0]
            )

        if not 0 <= value < 10000:
            print("Invalid CHIRPS value; skipping.")
            continue

        difference = abs(nasa_rain - value)

        results.append({
            "date": date_text,
            "location": location,
            "category": event["category"],
            "nasa_power_mm": nasa_rain,
            "chirps_mm": value,
            "absolute_difference_mm": difference,
            "nasa_heavy": nasa_rain >= THRESHOLD_MM,
            "chirps_heavy": value >= THRESHOLD_MM,
        })

        print(f"CHIRPS: {value:.2f} mm | Difference: {difference:.2f} mm")

    except Exception as error:
        print("Could not check this record:", error)

    finally:
        # Remove temporary downloads after each record
        for path in (gz_file, tif_file):
            if os.path.exists(path):
                os.remove(path)

if results:
    report = pd.DataFrame(results)
    report.to_csv(OUTPUT_FILE, index=False)

    print("\n=== SUMMARY ===")
    print("Successful comparisons:", len(report))
    print("\nResults by rainfall category:")
    print(
        report.groupby("category", observed=True)
        ["absolute_difference_mm"]
        .agg(["count", "mean", "median", "max"])
        .round(2)
    )

    print(
        "\nOverall mean absolute difference:",
        round(report["absolute_difference_mm"].mean(), 2),
        "mm",
    )

    disagreements = (
        report["nasa_heavy"] != report["chirps_heavy"]
    ).sum()

    print("Heavy-rain classification disagreements:", int(disagreements))
    print("Report saved to:", OUTPUT_FILE)

else:
    print("\nNo valid comparisons were completed.")
