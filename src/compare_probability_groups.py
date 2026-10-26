
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

# Load test data and trained model
test = pd.read_csv("data/test.csv")
model = joblib.load("models/heavy_rain_model.pkl")
features = joblib.load("models/model_features.pkl")

X_test = test[features]
y_test = test["heavy_rain_tomorrow"]

# Predict probability of heavy rainfall
probabilities = model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    "actual_heavy_rain": y_test,
    "probability": probabilities
})

# Compare probability distributions
print("\n--- PROBABILITY BY ACTUAL CLASS ---")

print(
    results.groupby("actual_heavy_rain")["probability"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(4)
    .to_string()
)

# Evaluate ranking performance
print("\n--- PROBABILITY RANKING METRICS ---")

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, probabilities), 4)
)

print(
    "Average precision:",
    round(average_precision_score(y_test, probabilities), 4)
)
