
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "forecast_training_data.csv"
MODEL_FILE = BASE_DIR / "models" / "forecast_heavy_rain_model.pkl"
FEATURE_FILE = BASE_DIR / "models" / "forecast_model_features.pkl"


# 2. Load dataset
print("Loading forecast dataset...")

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

# Add the same month feature used during training
df["month"] = df["date"].dt.month


# 3. Load feature names and trained model
feature_columns = joblib.load(FEATURE_FILE)
model = joblib.load(MODEL_FILE)

# Check that the required features exist
missing_features = [
    column for column in feature_columns
    if column not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features in dataset: {missing_features}"
    )

X = df[feature_columns]
y = df["heavy_rain_tomorrow"]


# 4. Recreate the 60% / 20% / 20% chronological split
unique_dates = sorted(df["date"].unique())

train_end_index = int(len(unique_dates) * 0.60)
validation_end_index = int(len(unique_dates) * 0.80)

validation_start_date = unique_dates[train_end_index]
test_start_date = unique_dates[validation_end_index]

validation_mask = (
    (df["date"] >= validation_start_date)
    & (df["date"] < test_start_date)
)

X_validation = X.loc[validation_mask]
y_validation = y.loc[validation_mask]


# 5. Generate probabilities for validation data
probabilities = model.predict_proba(X_validation)[:, 1]

print("\n--- VALIDATION DATA ---")
print("Features used:", feature_columns)
print("Validation starts:", validation_start_date.date())
print("Final test starts:", test_start_date.date())
print("Validation records:", len(X_validation))
print("Actual heavy-rain cases:", int(y_validation.sum()))

if y_validation.sum() == 0:
    raise ValueError(
        "The validation period contains no heavy-rain cases. "
        "Review the date split before continuing."
    )


# 6. Compare thresholds using validation data only
print("\n--- VALIDATION THRESHOLD COMPARISON ---")

thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    ).ravel()

    print(f"\nThreshold: {threshold:.2f}")
    print("Predicted heavy-rain cases:", int(predictions.sum()))
    print(f"True positives: {tp}")
    print(f"False alarms: {fp}")
    print(f"Missed heavy-rain cases: {fn}")
    print(f"True negatives: {tn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")


# 7. Finish
print("\nThreshold evaluation completed.")
print("These results use validation data, not the final test set.")
