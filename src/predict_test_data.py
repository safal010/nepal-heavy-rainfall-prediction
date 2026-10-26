
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

# Load test data and trained model
test = pd.read_csv("data/test.csv")
model = joblib.load("models/rainfall_model.pkl")
features = joblib.load("models/model_features.pkl")

# Prepare test features
X_test = test[features]

# Make predictions
predictions = model.predict(X_test)

# Create a results table
results = pd.DataFrame({
    "date": test["date"],
    "location": test["location"],
    "actual_rainfall_mm": test["rainfall_tomorrow_mm"],
    "predicted_rainfall_mm": predictions,
})

# Calculate absolute prediction errors
results["absolute_error_mm"] = (
    results["actual_rainfall_mm"]
    - results["predicted_rainfall_mm"]
).abs()

# Save predictions
results.to_csv("data/rainfall_predictions.csv", index=False)

print("Predictions saved to data/rainfall_predictions.csv")

print("\n--- FIRST 10 PREDICTIONS ---")
print(results.head(10).round(2).to_string(index=False))

mae = mean_absolute_error(
    results["actual_rainfall_mm"],
    results["predicted_rainfall_mm"]
)
print(f"\nOverall MAE: {mae:.2f} mm")

# Show the 10 largest prediction errors
print("\n--- TOP 10 LARGEST PREDICTION ERRORS ---")

largest_errors = results.sort_values(
    by="absolute_error_mm",
    ascending=False
).head(10)

print(largest_errors.round(2).to_string(index=False))
