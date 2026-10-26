
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "forecast_training_data.csv"
FEATURE_FILE = BASE_DIR / "models" / "forecast_model_features.pkl"
RAIN_MODEL_FILE = BASE_DIR / "models" / "forecast_rainfall_model.pkl"
HEAVY_MODEL_FILE = BASE_DIR / "models" / "forecast_heavy_rain_model.pkl"

# Load dataset and models
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

features = joblib.load(FEATURE_FILE)
rain_model = joblib.load(RAIN_MODEL_FILE)
heavy_model = joblib.load(HEAVY_MODEL_FILE)

# Recreate the chronological test split
unique_dates = sorted(df["date"].unique())
split_index = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_index]

test = df.loc[df["date"] >= split_date].copy()

X_test = test[features]

# Generate predictions
test["predicted_rainfall_mm"] = rain_model.predict(X_test)
test["heavy_rain_probability"] = heavy_model.predict_proba(X_test)[:, 1]

# Candidate threshold from our previous experiment
threshold = 0.20
test["predicted_heavy_rain"] = (
    test["heavy_rain_probability"] >= threshold
).astype(int)

test["rainfall_absolute_error_mm"] = (
    test["actual_rainfall_mm"] - test["predicted_rainfall_mm"]
).abs()

print("--- TEST PERIOD ---")
print("Start date:", test["date"].min().date())
print("End date:", test["date"].max().date())
print("Test records:", len(test))
print("Actual heavy-rain cases:", int(test["heavy_rain_tomorrow"].sum()))
print("Threshold:", threshold)

print("\n--- RAINFALL AMOUNT ERROR ---")
print(
    "MAE:",
    round(
        mean_absolute_error(
            test["actual_rainfall_mm"],
            test["predicted_rainfall_mm"],
        ),
        3,
    ),
    "mm",
)

# Correctly detected heavy-rain cases
detected = test.loc[
    (test["heavy_rain_tomorrow"] == 1)
    & (test["predicted_heavy_rain"] == 1)
]

print("\n--- HEAVY RAIN DETECTED CORRECTLY ---")
print("Count:", len(detected))

if detected.empty:
    print("No heavy-rain cases were detected at this threshold.")
else:
    print(
        detected[
            [
                "date",
                "location",
                "actual_rainfall_mm",
                "predicted_rainfall_mm",
                "heavy_rain_probability",
            ]
        ].to_string(index=False)
    )

# Missed heavy-rain cases
missed = test.loc[
    (test["heavy_rain_tomorrow"] == 1)
    & (test["predicted_heavy_rain"] == 0)
]

print("\n--- MISSED HEAVY-RAIN CASES ---")
print("Count:", len(missed))

if missed.empty:
    print("No heavy-rain cases were missed.")
else:
    print(
        missed[
            [
                "date",
                "location",
                "actual_rainfall_mm",
                "predicted_rainfall_mm",
                "heavy_rain_probability",
            ]
        ].to_string(index=False)
    )

# False alarms
false_alarms = test.loc[
    (test["heavy_rain_tomorrow"] == 0)
    & (test["predicted_heavy_rain"] == 1)
]

print("\n--- FALSE ALARMS ---")
print("Count:", len(false_alarms))

if false_alarms.empty:
    print("No false alarms.")
else:
    print(
        false_alarms[
            [
                "date",
                "location",
                "actual_rainfall_mm",
                "predicted_rainfall_mm",
                "heavy_rain_probability",
            ]
        ].to_string(index=False)
    )

# Largest rainfall prediction errors
print("\n--- 10 LARGEST RAINFALL ERRORS ---")

largest_errors = test.nlargest(10, "rainfall_absolute_error_mm")

print(
    largest_errors[
        [
            "date",
            "location",
            "actual_rainfall_mm",
            "predicted_rainfall_mm",
            "rainfall_absolute_error_mm",
            "heavy_rain_probability",
        ]
    ].to_string(index=False)
)
