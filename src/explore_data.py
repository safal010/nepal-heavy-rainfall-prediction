
import pandas as pd
from pathlib import Path

# Load our downloaded dataset
file_path = Path("data/nepal_weather.csv")

df = pd.read_csv(file_path)

# 1. Basic information
print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

# 2. Column names and data types
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

# 3. Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 4. Check duplicate rows
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

# 5. Summary statistics
print("\n--- SUMMARY STATISTICS ---")
print(df.describe())

# 6. Check available locations
print("\n--- LOCATIONS ---")
print(df["location"].value_counts())

# 7. Check date range
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("\n--- DATE RANGE ---")
print("Start:", df["date"].min())
print("End:", df["date"].max())

# 8. Inspect precipitation
if "PRECTOTCORR" in df.columns:
    print("\n--- PRECIPITATION SUMMARY ---")
    print(df["PRECTOTCORR"].describe())

    print("\n--- PRECIPITATION EXAMPLES ---")
    print(
        df[["date", "location", "PRECTOTCORR"]].head(10)
    )
