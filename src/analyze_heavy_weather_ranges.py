
import pandas as pd

# Load prepared data
df = pd.read_csv("data/prepared_weather.csv")

# Keep days followed by heavy rainfall
heavy = df[df["heavy_rain_tomorrow"] == 1]

features = [
    "T2M",
    "RH2M",
    "PS",
    "WS10M",
    "PRECTOTCORR",
    "rainfall_1_day_ago_mm",
    "rainfall_2_day_ago_mm",
    "rainfall_3_day_ago_mm",
]

print("\n--- WEATHER RANGES BEFORE HEAVY RAINFALL ---")
print(heavy[features].agg(["min", "median", "max"]).round(2).to_string())

# Inspect the Kathmandu example
case = df[
    (df["date"] == "2025-02-28")
    & (df["location"] == "Kathmandu")
]

if not case.empty:
    row = case.iloc[0]

    print("\n--- KATHMANDU EXTREME CASE ---")
    print(row[features].round(2).to_string())

