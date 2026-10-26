
import requests
import pandas as pd

url = "https://previous-runs-api.open-meteo.com/v1/forecast"

params = {
    "latitude": 27.7172,
    "longitude": 85.3240,
    "start_date": "2025-06-01",
    "end_date": "2025-06-30",
    "hourly": [
        "temperature_2m_previous_day1",
        "relative_humidity_2m_previous_day1",
        "precipitation_previous_day1",
        "wind_speed_10m_previous_day1",
    ],
    "timezone": "Asia/Kathmandu",
}

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

data = response.json()
df = pd.DataFrame(data["hourly"])

variables = [
    "temperature_2m_previous_day1",
    "relative_humidity_2m_previous_day1",
    "precipitation_previous_day1",
    "wind_speed_10m_previous_day1",
]

print("Hourly records:", len(df))
print("Expected hourly records:", 30 * 24)

df["date"] = pd.to_datetime(df["time"]).dt.date

print("\nValid hourly values:")
for variable in variables:
    print(f"{variable}: {df[variable].notna().sum()} / {len(df)}")

print("\nDaily coverage:")
daily = df.groupby("date")[variables].agg(
    lambda column: column.notna().sum()
)

print(daily.to_string())

print("\nDays with complete data:")
complete_days = daily.eq(24).all(axis=1)
print(f"{complete_days.sum()} / {len(daily)}")
