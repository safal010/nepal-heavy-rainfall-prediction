
import requests
import pandas as pd

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

EVENTS = [
    ("Biratnagar", 26.4525, 87.2718, "2025-09-13"),
    ("Jumla", 29.2747, 82.1838, "2022-10-07"),
    ("Kathmandu", 27.7172, 85.3240, "2024-09-24"),
    ("Nepalgunj", 28.0500, 81.6167, "2020-06-20"),
    ("Pokhara", 28.2096, 83.9856, "2023-06-30"),
]

def get_rainfall(latitude, longitude, date, time_standard):
    params = {
        "parameters": "PRECTOTCORR",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": date.replace("-", ""),
        "end": date.replace("-", ""),
        "format": "JSON",
        "time-standard": time_standard,
    }

    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()["properties"]["parameter"]["PRECTOTCORR"]
    return data.get(date.replace("-", ""))

results = []

for location, latitude, longitude, date in EVENTS:
    print(f"\n{location} | {date}")

    try:
        utc_value = get_rainfall(
            latitude, longitude, date, "UTC"
        )
        lst_value = get_rainfall(
            latitude, longitude, date, "LST"
        )

        results.append({
            "location": location,
            "date": date,
            "NASA_UTC_mm": utc_value,
            "NASA_LST_mm": lst_value,
        })

        print(f"UTC daily rainfall: {utc_value} mm")
        print(f"LST daily rainfall: {lst_value} mm")

        if utc_value is not None and lst_value is not None:
            print(
                "Absolute difference:",
                round(abs(float(utc_value) - float(lst_value)), 2),
                "mm",
            )

    except Exception as error:
        print("Could not retrieve data:", error)

if results:
    report = pd.DataFrame(results)
    output_file = "data/chirps/nasa_utc_vs_lst_check.csv"
    report.to_csv(output_file, index=False)
    print(f"\nComparison saved to: {output_file}")
