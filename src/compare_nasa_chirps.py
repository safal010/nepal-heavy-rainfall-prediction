
import os
import gzip
import requests
import pandas as pd
import rasterio
from rasterio.windows import Window

DATA_FILE = "data/nepal_weather.csv"
OUTPUT_FILE = "data/chirps/nasa_chirps_comparison.csv"
THRESHOLD_MM = 50.0

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

# Select the seven highest rainfall events, excluding
# the three events already checked.
already_checked = {
    ("Kathmandu", "2025-03-01"),
    ("Nepalgunj", "2025-06-11"),
    ("Pokhara", "2024-09-27"),
}

df["event_key"] = list(zip(
    df["location"],
    df["date"].dt.strftime("%Y-%m-%d")
))

candidates = df[~df["event_key"].isin(already_checked)]
events = candidates.nlargest(7, "PRECTOTCORR")

os.makedirs("data/chirps", exist_ok=True)

results = []

for _, event in events.iterrows():
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
        "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
        f"global_daily/tifs/p05/{year}/"
        f"chirps-v2.0.{year}.{month}.{day}.tif.gz"
    )

    gz_file = f"data/chirps/temp_{year}{month}{day}.tif.gz"
    tif_file = f"data/chirps/temp_{year}{month}{day}.tif"

    print(f"\nChecking {location} on {date_text}...")

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        with open(gz_file, "wb") as f:
            f.write(response.content)

        with gzip.open(gz_file, "rb") as source:
            with open(tif_file, "wb") as destination:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)

        
        with rasterio.open(tif_file) as dataset:
            row, col = dataset.index(longitude, latitude)

            # Read the CHIRPS grid cell containing the location.
            chirps_rain = float(
                dataset.read(1, window=Window(col, row, 1, 1))[0, 0]
            )

            # Read a 3x3 area around that grid cell.
            row_start = max(0, row - 1)
            col_start = max(0, col - 1)
            height = min(3, dataset.height - row_start)
            width = min(3, dataset.width - col_start)

            nearby = dataset.read(
                1,
                window=Window(col_start, row_start, width, height)
            )

            # Exclude invalid or missing-data values.
            valid_nearby = nearby[
                (nearby >= 0) & (nearby < 10000)
            ]

            if valid_nearby.size > 0:
                nearby_mean = float(valid_nearby.mean())
                nearby_min = float(valid_nearby.min())
                nearby_max = float(valid_nearby.max())
            else:
                nearby_mean = float("nan")
                nearby_min = float("nan")
                nearby_max = float("nan")


        # CHIRPS uses negative fill values for missing data.
        if not 0 <= chirps_rain < 10000:
            print("Invalid or missing CHIRPS value; skipping.")
            continue

        difference = abs(nasa_rain - chirps_rain)

        results.append({
            "date": date_text,
            "location": location,
            "nasa_power_mm": nasa_rain,
            "chirps_mm": chirps_rain,
            "absolute_difference_mm": difference,
            "nasa_heavy": nasa_rain >= THRESHOLD_MM,
            "chirps_heavy": chirps_rain >= THRESHOLD_MM,
            "chirps_nearby_mean_mm": nearby_mean,
            "chirps_nearby_min_mm": nearby_min,
            "chirps_nearby_max_mm": nearby_max,
        })

        print(f"NASA POWER: {nasa_rain:.2f} mm")
        print(f"CHIRPS:     {chirps_rain:.2f} mm")
        print(f"Difference: {difference:.2f} mm")

    except Exception as error:
        print("Could not check this event:", error)

if results:
    report = pd.DataFrame(results)
    report.to_csv(OUTPUT_FILE, index=False)

    print("\n=== COMPARISON SUMMARY ===")
    print(report.to_string(index=False))
    print(
        "\nMean absolute difference:",
        f"{report['absolute_difference_mm'].mean():.2f} mm"
    )
    print(
        "Heavy-rain classification disagreements:",
        int((report["nasa_heavy"] != report["chirps_heavy"]).sum())
    )
    print(f"\nReport saved to: {OUTPUT_FILE}")
else:
    print("\nNo valid comparisons were completed.")
