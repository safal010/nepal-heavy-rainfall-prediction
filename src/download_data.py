
import requests
import pandas as pd
from pathlib import Path

# Project data folder
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Approximate coordinates of six Nepal locations
locations = {
    "Kathmandu": (27.7172, 85.3240),
    "Pokhara": (28.2096, 83.9856),
    "Biratnagar": (26.4525, 87.2718),
    "Nepalgunj": (28.0500, 81.6167),
    "Janakpur": (26.7271, 85.9407),
    "Jumla": (29.2747, 82.1838),
}

url = (
    "https://power.larc.nasa.gov/api/temporal/"
    "daily/point"
)

all_data = []

for city, (latitude, longitude) in locations.items():
    print(f"Downloading weather data for {city}...")

    params = {
        "parameters": (
            "T2M,RH2M,PS,WS10M,PRECTOTCORR"
        ),
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": "20150101",
        "end": "20251231",
        "format": "JSON",
        "time-standard": "UTC",
    }

    response = requests.get(
        url,
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()
    weather = result["properties"]["parameter"]

    df = pd.DataFrame(weather)
    df.index.name = "date"
    df = df.reset_index()

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y%m%d",
        errors="coerce",
    )

    df["location"] = city
    df["latitude"] = latitude
    df["longitude"] = longitude

    # Convert documented missing-data markers to NaN.
    df = df.replace(-999.0, float("nan"))

    all_data.append(df)

# Combine all locations into one dataset
final_df = pd.concat(
    all_data,
    ignore_index=True,
)

output_file = DATA_DIR / "nepal_weather.csv"
final_df.to_csv(output_file, index=False)

print("\nDownload completed!")
print("Dataset shape:", final_df.shape)
print("Columns:", final_df.columns.tolist())
print("Saved to:", output_file.resolve())

print("\nFirst five rows:")
print(final_df.head())
