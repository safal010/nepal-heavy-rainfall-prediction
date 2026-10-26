
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FILE = BASE_DIR / "data" / "forecast_training_data.csv"

df = pd.read_csv(FILE)

print("--- DATASET SHAPE ---")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n--- DATE COVERAGE ---")
print("Earliest date:", df["date"].min())
print("Latest date:", df["date"].max())

print("\n--- ROWS PER LOCATION ---")
print(df.groupby("location").size().to_string())

print("\n--- DUPLICATE DATE/LOCATION RECORDS ---")
duplicates = df.duplicated(subset=["date", "location"]).sum()
print("Duplicates:", duplicates)

print("\n--- MISSING VALUES ---")
print(df.isna().sum().to_string())

print("\n--- HEAVY-RAIN LABEL CHECK ---")
expected_label = (
    df["actual_rainfall_mm"] >= 50.0
).astype(int)

print(
    "Labels match the 50 mm threshold:",
    expected_label.equals(df["heavy_rain_tomorrow"])
)

print("\n--- RAINFALL TARGET SUMMARY ---")
print(df["actual_rainfall_mm"].describe().to_string())

print("\n--- HEAVY-RAIN CASES BY LOCATION ---")
print(
    df.groupby("location")["heavy_rain_tomorrow"]
    .agg(["count", "sum"])
    .to_string()
)

print("\n--- DAILY DATE GAPS BY LOCATION ---")
for location, group in df.groupby("location"):
    dates = pd.to_datetime(group["date"]).sort_values()
    gaps = dates.diff().dt.days
    missing_days = int((gaps - 1).clip(lower=0).sum())

    print(
        f"{location}: {missing_days} missing calendar days "
        f"between retained records"
    )


