
import pandas as pd
import joblib

from sklearn.metrics import classification_report

# Load test data and trained model
test = pd.read_csv("data/test.csv")
model = joblib.load("models/heavy_rain_model.pkl")

X_test = test.drop(
    columns=["rainfall_tomorrow_mm", "heavy_rain_tomorrow"]
)
y_test = test["heavy_rain_tomorrow"]

# Predict probabilities and apply our experimental threshold
heavy_index = list(model.classes_).index(1)
probabilities = model.predict_proba(X_test)[:, heavy_index]

threshold = 0.05
predictions = (probabilities >= threshold).astype(int)

# Show detailed metrics
print("\n--- CLASSIFICATION REPORT ---")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No heavy rain", "Heavy rain"],
        zero_division=0,
    )
)

# Add predictions to the test data for analysis
results = test.copy()
results["heavy_rain_probability"] = probabilities
results["predicted_heavy_rain"] = predictions

# Convert the encoded location columns back into a readable location
location_columns = [
    column for column in X_test.columns
    if column.startswith("location_")
]

if location_columns:
    results["location"] = (
        X_test[location_columns]
        .idxmax(axis=1)
        .str.replace("location_", "", regex=False)
    )

# Separate correct detections, false alarms, and missed events
true_positives = results[
    (results["heavy_rain_tomorrow"] == 1)
    & (results["predicted_heavy_rain"] == 1)
]

false_positives = results[
    (results["heavy_rain_tomorrow"] == 0)
    & (results["predicted_heavy_rain"] == 1)
]

false_negatives = results[
    (results["heavy_rain_tomorrow"] == 1)
    & (results["predicted_heavy_rain"] == 0)
]

print("\nCorrectly detected heavy-rain events:", len(true_positives))
print("False alarms:", len(false_positives))
print("Missed heavy-rain events:", len(false_negatives))


columns_to_show = [
    "location",
    "PRECTOTCORR",
    "rainfall_tomorrow_mm",
    "heavy_rain_probability"
]

print(false_negatives[columns_to_show].head(10).to_string(index=False))


# Save the full prediction results
results.to_csv("data/heavy_rain_predictions.csv", index=False)
print("\nSaved predictions to data/heavy_rain_predictions.csv")
