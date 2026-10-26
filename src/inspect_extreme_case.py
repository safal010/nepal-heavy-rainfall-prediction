
import pandas as pd

# Load the prepared weather data
df = pd.read_csv("data/prepared_weather.csv")

# Inspect the input date and the following day's rainfall
target_date = "2025-02-28"
target_location = "Kathmandu"

case = df[
    (df["date"] == target_date)
    & (df["location"] == target_location)
]

if case.empty:
    print("No matching record found.")
else:
    row = case.iloc[0]

    print("\n--- EXTREME RAINFALL CASE ---")
    print("Location:", row["location"])
    print("Input date:", row["date"])
    print("Rainfall today:", row["PRECTOTCORR"], "mm")
    print("Rainfall 1 day ago:", row["rainfall_1_day_ago_mm"], "mm")
    print("Rainfall 2 days ago:", row["rainfall_2_day_ago_mm"], "mm")
    print("Rainfall 3 days ago:", row["rainfall_3_day_ago_mm"], "mm")
    print("Target rainfall tomorrow:", row["rainfall_tomorrow_mm"], "mm")
