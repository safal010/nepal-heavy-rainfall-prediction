
import requests
import pandas as pd

url = "https://power.larc.nasa.gov/api/temporal/daily/point"

params = {
    "parameters": "PRECTOTCORR",
    "community": "RE",
    "longitude": 85.3240,
    "latitude": 27.7172,
    "start": "20250830",
    "end": "20250906",
    "format": "JSON",
}

for time_standard in ["UTC", "LST"]:
    params["time-standard"] = time_standard

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    result = response.json()
    rainfall = result["properties"]["parameter"]["PRECTOTCORR"]

    df = pd.DataFrame(
        list(rainfall.items()),
        columns=["date", "rainfall_mm"],
    )

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y%m%d",
    )

    print(f"\n--- NASA POWER: {time_standard} ---")
    print(df.to_string(index=False))
