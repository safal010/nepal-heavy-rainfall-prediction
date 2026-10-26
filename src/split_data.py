
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Load prepared dataset
# --------------------------------------------------

df = pd.read_csv("data/prepared_weather.csv")
df["date"] = pd.to_datetime(df["date"])

# Sort chronologically within each location
df = df.sort_values(
    ["date", "location"]
).reset_index(drop=True)

# --------------------------------------------------
# 2. Create calendar features
# --------------------------------------------------

df["day_of_year"] = df["date"].dt.dayofyear

# Represent seasonal patterns smoothly
df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.25
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.25
)

# --------------------------------------------------
# 3. Define model input features
# --------------------------------------------------

feature_columns = [
    "T2M",
    "RH2M",
    "PS",
    "WS10M",
    "PRECTOTCORR",
    "rainfall_1_day_ago_mm",
    "rainfall_2_day_ago_mm",
    "rainfall_3_day_ago_mm",
    "latitude",
    "longitude",
    "day_sin",
    "day_cos",
    "location",
]

target_columns = [
    "rainfall_tomorrow_mm",
    "heavy_rain_tomorrow",
]

# --------------------------------------------------
# 4. Remove rows with missing input features
# --------------------------------------------------

# The first few dates at each location may not have
# enough historical data to calculate all lag features.
df = df.dropna(
    subset=feature_columns
).copy()

# --------------------------------------------------
# 5. Split chronologically by date
# --------------------------------------------------

unique_dates = sorted(df["date"].unique())

split_index = int(len(unique_dates) * 0.80)
cutoff_date = unique_dates[split_index]

train = df[df["date"] < cutoff_date].copy()
test = df[df["date"] >= cutoff_date].copy()

# --------------------------------------------------
# 6. Encode location for machine learning
# --------------------------------------------------

X_train = pd.get_dummies(
    train[feature_columns],
    columns=["location"],
    dtype=int,
)

X_test = pd.get_dummies(
    test[feature_columns],
    columns=["location"],
    dtype=int,
)

# Ensure both datasets have identical feature columns
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0,
)

# --------------------------------------------------
# 7. Save features, targets, and metadata
# --------------------------------------------------

# Keep dates and location names for later analysis.
# They are not included in the model input features.
metadata_columns = ["date", "location"]

train_output = train[metadata_columns].reset_index(drop=True)
test_output = test[metadata_columns].reset_index(drop=True)

train_output = pd.concat(
    [train_output, X_train.reset_index(drop=True)],
    axis=1,
)

test_output = pd.concat(
    [test_output, X_test.reset_index(drop=True)],
    axis=1,
)

for target in target_columns:
    train_output[target] = train[target].to_numpy()
    test_output[target] = test[target].to_numpy()

Path("data").mkdir(exist_ok=True)

train_output.to_csv("data/train.csv", index=False)
test_output.to_csv("data/test.csv", index=False)

# --------------------------------------------------
# 8. Display split information
# --------------------------------------------------

print("\n--- CHRONOLOGICAL SPLIT COMPLETE ---")

print(
    "Training date range:",
    train["date"].min().date(),
    "to",
    train["date"].max().date(),
)

print(
    "Testing date range:",
    test["date"].min().date(),
    "to",
    test["date"].max().date(),
)

print("\nTraining rows:", len(train_output))
print("Testing rows:", len(test_output))
print("Number of model input features:", len(X_train.columns))

print("\n--- HEAVY-RAINFALL COUNTS ---")
print("Training:")
print(train["heavy_rain_tomorrow"].value_counts().sort_index())

print("\nTesting:")
print(test["heavy_rain_tomorrow"].value_counts().sort_index())

print("\n--- NEW RAINFALL FEATURES ---")
print(
    [
        "rainfall_1_day_ago_mm",
        "rainfall_2_day_ago_mm",
        "rainfall_3_day_ago_mm",
    ]
)

print("\n--- FIRST 3 TEST ROWS ---")
print(test_output.head(3).to_string(index=False))

print("\nSaved data/train.csv and data/test.csv")
