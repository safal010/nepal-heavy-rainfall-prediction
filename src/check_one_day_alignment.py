
import requests
import pandas as pd

latitude = 27.7172
longitude = 85.3240
target_date = "2025-09-02"

url = "https://single-runs-api.open-meteo.com/v1/forecast"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": "precipitation",
    "timezone": "UTC",
    "models": "ecmwf_ifs",
    "run": "2025-09-01T00:00",
    "forecast_hours": 96,
}

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

hourly = response.json()["hourly"]
df = pd.DataFrame({
    "time_utc": pd.to_datetime(hourly["time"], utc=True),
    "rainfall_mm": hourly["precipitation"],
})

df = df.dropna(subset=["rainfall_mm"])
df["time_nepal"] = df["time_utc"].dt.tz_convert("Asia/Kathmandu")

utc_day = df[df["time_utc"].dt.strftime("%Y-%m-%d") == target_date]
nepal_day = df[df["time_nepal"].dt.strftime("%Y-%m-%d") == target_date]

print("Target date:", target_date)
print("\nUTC day")
print("Valid hours:", len(utc_day))
print("Rainfall total:", round(utc_day["rainfall_mm"].sum(), 2), "mm")

print("\nNepal local day")
print("Valid hours:", len(nepal_day))
print("Rainfall total:", round(nepal_day["rainfall_mm"].sum(), 2), "mm")

observations = pd.read_csv("data/nepal_weather.csv")
observations["date"] = pd.to_datetime(observations["date"]).dt.strftime("%Y-%m-%d")

row = observations[
    (observations["location"] == "Kathmandu")
    & (observations["date"] == target_date)
]

print("\nNASA POWER")
if row.empty:
    print("No observation found.")
else:
    print("Rainfall:", row["PRECTOTCORR"].iloc[0], "mm")
