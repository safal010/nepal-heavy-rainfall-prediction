
import pandas as pd

# Load the prepared weather data
df = pd.read_csv("data/prepared_weather.csv")

# Select the extreme rainfall case
case = df[
    (df["date"] == "2025-02-28")
    & (df["location"] == "Kathmandu")
]

if case.empty:
    print("No matching record found.")
else:
    row = case.iloc[0]

    print("\n--- WEATHER CONDITIONS BEFORE EXTREME RAINFALL ---")
    print("Location:", row["location"])
    print("Input date:", row["date"])
    print("Temperature (T2M):", row["T2M"], "°C")
    print("Relative humidity (RH2M):", row["RH2M"], "%")
    print("Surface pressure (PS):", row["PS"], "kPa or dataset units")
    print("Wind speed (WS10M):", row["WS10M"], "m/s")
    print("Rainfall (PRECTOTCORR):", row["PRECTOTCORR"], "mm")
    print("Target rainfall tomorrow:", row["rainfall_tomorrow_mm"], "mm")
