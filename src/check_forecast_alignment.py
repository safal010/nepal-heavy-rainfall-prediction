
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "forecast_training_data.csv"

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

print("--- SAMPLE DATASET ROWS ---")
print(
    df[
        [
            "date",
            "location",
            "forecast_rainfall_mm",
            "actual_rainfall_mm",
            "heavy_rain_tomorrow",
        ]
    ].head(15).to_string(index=False)
)

print("\n--- HEAVY-RAIN EVENTS ---")
heavy = df[df["actual_rainfall_mm"] >= 50]

print("Total heavy-rain events:", len(heavy))
print(
    heavy[
        [
            "date",
            "location",
            "forecast_rainfall_mm",
            "actual_rainfall_mm",
        ]
    ].head(20).to_string(index=False)
)

print("\n--- FORECAST RAINFALL VS ACTUAL RAINFALL ---")
print(
    df[
        ["forecast_rainfall_mm", "actual_rainfall_mm"]
    ].describe().round(2).to_string()
)

print("\n--- FORECAST RAINFALL ON HEAVY-RAIN DAYS ---")
print(
    df.loc[
        df["actual_rainfall_mm"] >= 50,
        "forecast_rainfall_mm",
    ].describe().round(2).to_string()
)

print("\n--- CHECK DATE LABELING ---")
print(
    "Each row contains forecast features and observed rainfall "
    "for the same calendar date."
)
print(
    "The API's previous_day1 suffix indicates a 24-hour lead "
    "time, but this script alone cannot verify the exact forecast "
    "issue timestamp or the model's publication time."
)
