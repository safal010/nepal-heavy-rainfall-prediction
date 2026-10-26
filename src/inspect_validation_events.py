
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import average_precision_score

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "forecast_training_data.csv"
MODEL_FILE = BASE_DIR / "models" / "forecast_heavy_rain_model.pkl"
FEATURE_FILE = BASE_DIR / "models" / "forecast_model_features.pkl"

# 2. Load the data
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month

# 3. Load the trained model and features
model = joblib.load(MODEL_FILE)
feature_columns = joblib.load(FEATURE_FILE)

X = df[feature_columns]
y = df["heavy_rain_tomorrow"]

# 4. Recreate the chronological validation split
unique_dates = sorted(df["date"].unique())

train_end_index = int(len(unique_dates) * 0.60)
validation_end_index = int(len(unique_dates) * 0.80)

validation_start = unique_dates[train_end_index]
test_start = unique_dates[validation_end_index]

validation_mask = (
    (df["date"] >= validation_start)
    & (df["date"] < test_start)
)

validation_df = df.loc[validation_mask].copy()
X_validation = X.loc[validation_mask]
y_validation = y.loc[validation_mask]

# 5. Calculate predicted probabilities
probabilities = model.predict_proba(X_validation)[:, 1]
validation_df["heavy_rain_probability"] = probabilities

print("Validation starts:", validation_start)
print("Final test starts:", test_start)
print("Validation records:", len(validation_df))
print("Actual heavy-rain cases:", int(y_validation.sum()))

# Average precision summarizes the precision-recall curve.
# With only three positive validation cases, interpret cautiously.
print(
    "Average precision:",
    round(average_precision_score(y_validation, probabilities), 4),
)

# 6. Show actual heavy-rain cases
print("\n--- ACTUAL HEAVY-RAIN CASES ---")

heavy_cases = validation_df[
    validation_df["heavy_rain_tomorrow"] == 1
][
    [
        "date",
        "location",
        "forecast_rainfall_mm",
        "humidity_mean_pct",
        "actual_rainfall_mm",
        "heavy_rain_probability",
    ]
].sort_values("heavy_rain_probability", ascending=False)

print(heavy_cases.to_string(index=False))

# 7. Show the ten highest predicted probabilities
print("\n--- TOP 10 PREDICTED HEAVY-RAIN PROBABILITIES ---")

top_predictions = validation_df.sort_values(
    "heavy_rain_probability",
    ascending=False,
).head(10)

print(
    top_predictions[
        [
            "date",
            "location",
            "forecast_rainfall_mm",
            "humidity_mean_pct",
            "actual_rainfall_mm",
            "heavy_rain_tomorrow",
            "heavy_rain_probability",
        ]
    ].to_string(index=False)
)
