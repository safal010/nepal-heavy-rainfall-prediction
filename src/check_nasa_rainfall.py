
import requests
import pandas as pd

base_url = "https://power.larc.nasa.gov/api/temporal/"

params = {
    "parameters": "PRECTOTCORR",
    "community": "RE",
    "longitude": 87.2718,
    "latitude": 26.4525,
    "start": "20250227",
    "end": "20250303",
    "format": "JSON",
    "time-standard": "UTC",
}

hourly_response = requests.get(
    base_url + "hourly/point",
    params=params,
    timeout=60,
)
daily_response = requests.get(
    base_url + "daily/point",
    params=params,
    timeout=60,
)

hourly_response.raise_for_status()
daily_response.raise_for_status()

hourly_data = (
    hourly_response.json()["properties"]["parameter"]["PRECTOTCORR"]
)
daily_data = (
    daily_response.json()["properties"]["parameter"]["PRECTOTCORR"]
)

hourly = pd.Series(
    list(hourly_data.values()),
    index=[timestamp[:8] for timestamp in hourly_data],
)

# Exclude NASA POWER's missing-data marker and other negative values.
hourly = hourly[hourly >= 0]
hourly_totals = hourly.groupby(level=0).sum()

print("Date       Hourly sum (mm)   Daily API (mm)")
for day in sorted(daily_data):
    hourly_total = hourly_totals.get(day, float("nan"))
    daily_total = daily_data[day]

    print(f"{day}   {hourly_total:14.2f}   {daily_total:13.2f}")
