
import os
import gzip
import requests
import pandas as pd
import rasterio
from rasterio.windows import Window

# Load the location coordinates from your existing dataset
weather = pd.read_csv("data/nepal_weather.csv")

locations = (
    weather[["location", "latitude", "longitude"]]
    .drop_duplicates("location")
    .set_index("location")
)

# Events to investigate
checks = [
    ("Kathmandu", "2025-09-01"),
    ("Kathmandu", "2025-09-02"),
    ("Kathmandu", "2025-09-03"),
]

base_url = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    "global_daily/tifs/p05"
)

os.makedirs("data/chirps", exist_ok=True)

for location, date in checks:
    print(f"\n{'=' * 55}")
    print(f"{location} | {date}")
    print("=" * 55)

    latitude = locations.loc[location, "latitude"]
    longitude = locations.loc[location, "longitude"]

    date_parts = date.split("-")
    year, month, day = date_parts

    url = (
        f"{base_url}/{year}/"
        f"chirps-v2.0.{year}.{month}.{day}.tif.gz"
    )

    gz_file = f"data/chirps/{location}_{date}.tif.gz"
    tif_file = f"data/chirps/{location}_{date}.tif"

    try:
        print(f"Coordinates: {latitude}, {longitude}")
        print("Downloading CHIRPS map...")

        response = requests.get(url, timeout=120)
        print("HTTP status:", response.status_code)
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

            value = dataset.read(
                1, window=Window(col, row, 1, 1)
            )[0, 0]

            nearby = dataset.read(
                1,
                window=Window(col - 1, row - 1, 3, 3)
            )

            valid = nearby[(nearby >= 0) & (nearby < 10000)]

            print(f"CHIRPS at location: {value:.2f} mm/day")

            if len(valid):
                print(f"Nearby grid-cell mean: {valid.mean():.2f} mm/day")
                print(f"Nearby minimum: {valid.min():.2f} mm/day")
                print(f"Nearby maximum: {valid.max():.2f} mm/day")

    except Exception as error:
        print("Could not check this event:", error)

print("\nCHIRPS comparison finished.")
