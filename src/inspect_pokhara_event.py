
import requests
import pandas as pd

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def get_daily_rainfall(time_standard):
    params = {
        "parameters": "PRECTOTCORR",
        "community": "RE",
        "longitude": 83.9856,
        "latitude": 28.2096,
        "start": "20240729",
        "end": "20240731",
        "format": "JSON",
        "time-standard": time_standard,
    }

    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()

    values = (
        response.json()["properties"]["parameter"]["PRECTOTCORR"]
    )

    return {
        date: float(value)
        for date, value in values.items()
    }

utc = get_daily_rainfall("UTC")
lst = get_daily_rainfall("LST")

report = pd.DataFrame({
    "date": sorted(utc.keys()),
})

report["UTC_rainfall_mm"] = report["date"].map(utc)
report["LST_rainfall_mm"] = report["date"].map(lst)

print("\nPokhara rainfall: July 29–31, 2024")
print(report.to_string(index=False))

print("\nThree-day UTC total:", round(report["UTC_rainfall_mm"].sum(), 2), "mm")
print("Three-day LST total:", round(report["LST_rainfall_mm"].sum(), 2), "mm")
