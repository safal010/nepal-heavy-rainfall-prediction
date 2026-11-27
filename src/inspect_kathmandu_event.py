
import requests
import pandas as pd

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def get_daily_rainfall(time_standard):
    params = {
        "parameters": "PRECTOTCORR",
        "community": "RE",
        "longitude": 85.3240,
        "latitude": 27.7172,
        "start": "20240923",
        "end": "20240927",
        "format": "JSON",
        "time-standard": time_standard,
    }

    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()

    values = response.json()["properties"]["parameter"]["PRECTOTCORR"]

    return {
        date: float(value)
        for date, value in values.items()
    }

utc = get_daily_rainfall("UTC")
lst = get_daily_rainfall("LST")

dates = sorted(set(utc) & set(lst))

report = pd.DataFrame({
    "date": dates,
    "UTC_rainfall_mm": [utc[d] for d in dates],
    "LST_rainfall_mm": [lst[d] for d in dates],
})

print("\nKathmandu rainfall: September 23–27, 2024")
print(report.to_string(index=False))

print("\nFive-day UTC total:", round(report["UTC_rainfall_mm"].sum(), 2), "mm")
print("Five-day LST total:", round(report["LST_rainfall_mm"].sum(), 2), "mm")
