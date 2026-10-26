
import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA_FILE = "data/fixed_run_forecast_training_data.csv"
MODEL_FILE = "models/fixed_run_heavy_rain_model.pkl"
FEATURES_FILE = "models/fixed_run_model_features.pkl"

# Load the saved model and data
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

model = joblib.load(MODEL_FILE)
features = joblib.load(FEATURES_FILE)

# Use the same untouched 2025 test period
test = df[df["date"] >= "2025-01-01"].copy()

X_test = test[features]
y_test = test["heavy_rain_tomorrow"]

# Get predicted probabilities once
probabilities = model.predict_proba(X_test)[:, 1]

print("Test records:", len(test))
print("Actual heavy-rain cases:", int(y_test.sum()))
print("\nTHRESHOLD COMPARISON")
print("-" * 75)

for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f} | "
        f"Alerts: {tp + fp} | "
        f"Detected: {tp} | Missed: {fn} | "
        f"False alarms: {fp}"
    )
