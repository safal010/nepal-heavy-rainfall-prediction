
import pandas as pd
import joblib

# Load test data and the trained model
test = pd.read_csv("data/test.csv")
model = joblib.load("models/heavy_rain_model.pkl")
features = joblib.load("models/model_features.pkl")

# Separate input features from the actual answers
X_test = test[features]
y_test = test["heavy_rain_tomorrow"]

# Get the model's probability for heavy rainfall
probabilities = model.predict_proba(X_test)[:, 1]

# Combine actual results and predicted probabilities
results = test[["date", "location"]].copy()
results["actual_heavy_rain"] = y_test
results["predicted_probability"] = probabilities

# Separate actual heavy-rainfall cases
heavy_cases = results[results["actual_heavy_rain"] == 1]

print("\n--- PROBABILITIES FOR ACTUAL HEAVY-RAINFALL CASES ---")
print(
    heavy_cases["predicted_probability"]
    .describe()
    .round(4)
    .to_string()
)

print("\n--- HIGHEST-PROBABILITY ACTUAL HEAVY-RAINFALL CASES ---")
print(
    heavy_cases.sort_values(
        "predicted_probability", ascending=False
    ).head(10).to_string(index=False)
)

print("\n--- PROBABILITY GROUP COUNTS ---")
bins = [0, 0.05, 0.10, 0.20, 0.50, 1.0]

print(
    pd.cut(
        heavy_cases["predicted_probability"],
        bins=bins,
        include_lowest=True
    ).value_counts(sort=False).to_string()
)
