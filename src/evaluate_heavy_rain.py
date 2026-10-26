
import pandas as pd
import joblib

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# 1. Load testing data and trained model
# --------------------------------------------------

test = pd.read_csv("data/test.csv")
model = joblib.load("models/heavy_rain_model.pkl")

target = "heavy_rain_tomorrow"

# Remove target columns and metadata from model inputs
X_test = test.drop(
    columns=[
        "date",
        "location",
        "rainfall_tomorrow_mm",
        "heavy_rain_tomorrow",
    ],
)

y_test = test[target]

# --------------------------------------------------
# 2. Check the predicted probabilities
# --------------------------------------------------

heavy_class_index = list(model.classes_).index(1)

probabilities = model.predict_proba(X_test)[
    :, heavy_class_index
]

print("\n--- PROBABILITY INSPECTION ---")
print(
    "Highest predicted probability:",
    round(probabilities.max(), 4),
)
print(
    "Average predicted probability:",
    round(probabilities.mean(), 4),
)

# --------------------------------------------------
# 3. Compare different thresholds
# --------------------------------------------------

print("\n--- THRESHOLD COMPARISON ---")

for threshold in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Predicted heavy-rain cases: {predictions.sum()}")
    print(f"Correct detections: {tp}")
    print(f"False alarms: {fp}")
    print(f"Missed heavy-rain cases: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 score: {f1:.3f}")
    print(
        f"Confusion matrix: [[{tn}, {fp}], [{fn}, {tp}]]"
    )
