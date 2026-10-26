
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. File locations
# --------------------------------------------------

input_file = Path("data/nepal_weather.csv")
output_file = Path("data/prepared_weather.csv")

# --------------------------------------------------
# 2. Load the original dataset
# --------------------------------------------------

df = pd.read_csv(input_file)

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Check required columns
required_columns = [
    "date",
    "location",
    "T2M",
    "RH2M",
    "PS",
    "WS10M",
    "PRECTOTCORR",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Remove records with invalid dates
df = df.dropna(subset=["date"]).copy()

# Sort chronologically within each location
df = df.sort_values(
    ["location", "date"]
).reset_index(drop=True)

# Check for duplicate location-date records
duplicates = df.duplicated(
    subset=["location", "date"]
)

if duplicates.any():
    raise ValueError(
        "Duplicate date records found within a location."
    )

# --------------------------------------------------
# 3. Create historical rainfall features
# --------------------------------------------------

# Each feature uses rainfall from earlier days
for days in [1, 2, 3]:
    column_name = f"rainfall_{days}_day_ago_mm"

    df[column_name] = (
        df.groupby("location")["PRECTOTCORR"]
        .shift(days)
    )

    # Check whether the lag comes from the correct
    # calendar day, not merely the previous row.
    previous_date = (
        df.groupby("location")["date"]
        .shift(days)
    )

    valid_lag = (
        (df["date"] - previous_date).dt.days == days
    )

    df.loc[~valid_lag, column_name] = float("nan")

# --------------------------------------------------
# 4. Create tomorrow's rainfall target
# --------------------------------------------------

df["rainfall_tomorrow_mm"] = (
    df.groupby("location")["PRECTOTCORR"]
    .shift(-1)
)

# Find the date of the next record for each location
next_date = (
    df.groupby("location")["date"]
    .shift(-1)
)

# Ensure tomorrow's target is exactly the next day
consecutive_day = (
    (next_date - df["date"]).dt.days == 1
)

df.loc[
    ~consecutive_day,
    "rainfall_tomorrow_mm"
] = float("nan")

# --------------------------------------------------
# 5. Create the heavy-rain classification target
# --------------------------------------------------

# Provisional threshold: 50 mm in one day
HEAVY_RAIN_THRESHOLD_MM = 50.0

df["heavy_rain_tomorrow"] = (
    df["rainfall_tomorrow_mm"]
    .ge(HEAVY_RAIN_THRESHOLD_MM)
    .astype("Int64")
)

# --------------------------------------------------
# 6. Remove records without a valid target
# --------------------------------------------------

prepared = df.dropna(
    subset=["rainfall_tomorrow_mm"]
).copy()

# --------------------------------------------------
# 7. Save the prepared dataset
# --------------------------------------------------

prepared.to_csv(output_file, index=False)

print("\n--- PREPARATION COMPLETE ---")
print("Original rows:", len(df))
print("Prepared rows:", len(prepared))
print("Saved to:", output_file.resolve())

# --------------------------------------------------
# 8. Display sample records
# --------------------------------------------------

columns_to_show = [
    "date",
    "location",
    "PRECTOTCORR",
    "rainfall_1_day_ago_mm",
    "rainfall_2_day_ago_mm",
    "rainfall_3_day_ago_mm",
    "rainfall_tomorrow_mm",
    "heavy_rain_tomorrow",
]

print("\n--- FIRST 5 PREPARED ROWS ---")
print(
    prepared[columns_to_show]
    .head()
    .to_string(index=False)
)

# --------------------------------------------------
# 9. Check target distribution
# --------------------------------------------------

print("\n--- TARGET DISTRIBUTION ---")
print(
    prepared["heavy_rain_tomorrow"]
    .value_counts()
    .sort_index()
    .rename(
        index={
            0: "No heavy rain",
            1: "Heavy rain",
        }
    )
)

# --------------------------------------------------
# 10. Check missing values in new features
# --------------------------------------------------

columns_to_check = [
    "rainfall_1_day_ago_mm",
    "rainfall_2_day_ago_mm",
    "rainfall_3_day_ago_mm",
    "rainfall_tomorrow_mm",
    "heavy_rain_tomorrow",
]

print("\n--- MISSING VALUES ---")
print(prepared[columns_to_check].isna().sum())
