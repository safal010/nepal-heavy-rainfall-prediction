
import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Load test data and both trained models
test = pd.read_csv("data/test.csv")
features = joblib.load("models/model_features.pkl")

X_test = test[features]
y_test = test["heavy_rain_tomorrow"]

models = {
    "Random Forest": joblib.load("models/heavy_rain_model.pkl"),
    "Logistic Regression": joblib.load(
        "models/logistic_regression_heavy_rain.pkl"
    ),
}

thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]

rows = []

for model_name, model in models.items():
    probabilities = model.predict_proba(X_test)[:, 1]

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        rows.append({
            "model": model_name,
            "threshold": threshold,
            "predicted_heavy": int(predictions.sum()),
            "true_positives": int(tp),
            "false_alarms": int(fp),
            "missed_events": int(fn),
            "precision": round(
                precision_score(y_test, predictions, zero_division=0), 3
            ),
            "recall": round(
                recall_score(y_test, predictions, zero_division=0), 3
            ),
            "f1": round(
                f1_score(y_test, predictions, zero_division=0), 3
            ),
        })

results = pd.DataFrame(rows)

print("\n--- MODEL COMPARISON AT DIFFERENT THRESHOLDS ---")
print(results.to_string(index=False))

results.to_csv("data/model_threshold_comparison.csv", index=False)
print("\nSaved to data/model_threshold_comparison.csv")
