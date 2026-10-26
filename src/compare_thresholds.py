
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Load test data and trained classifier
test = pd.read_csv("data/test.csv")
model = joblib.load("models/heavy_rain_model.pkl")
features = joblib.load("models/model_features.pkl")

X_test = test[features]
y_actual = test["heavy_rain_tomorrow"]

# Get the probability of heavy rainfall
probabilities = model.predict_proba(X_test)[:, 1]

print("\n--- HEAVY-RAINFALL THRESHOLD COMPARISON ---")
print("Threshold | Precision | Recall | F1 Score | TP | FP | FN | TN")

for threshold in [0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_actual, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_actual, predictions, zero_division=0
    )
    recall = recall_score(
        y_actual, predictions, zero_division=0
    )
    f1 = f1_score(
        y_actual, predictions, zero_division=0
    )

    print(
        f"{threshold:9.2f} | {precision:9.3f} | "
        f"{recall:6.3f} | {f1:8.3f} | "
        f"{tp:2d} | {fp:2d} | {fn:2d} | {tn:4d}"
    )
