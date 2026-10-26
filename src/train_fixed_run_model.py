
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    average_precision_score,
)

# 1. Define file paths
DATA_FILE = Path("data/fixed_run_forecast_training_data.csv")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)

# 2. Load the dataset
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

print("Dataset loaded successfully")
print("Total records:", len(df))

# 3. Choose the input features
features = [
    "latitude",
    "longitude",
    "temperature_mean_c",
    "humidity_mean_pct",
    "forecast_rainfall_mm",
    "wind_speed_mean_kmh",
]

target = "heavy_rain_tomorrow"

# 4. Split chronologically: train on 2024, test on 2025
train = df[df["date"] < "2025-01-01"].copy()
test = df[df["date"] >= "2025-01-01"].copy()

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print("\nTraining records:", len(train))
print("Training heavy-rain cases:", int(y_train.sum()))

print("\nTesting records:", len(test))
print("Testing heavy-rain cases:", int(y_test.sum()))

# 5. Train the classifier
model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("\nModel training completed!")

# 6. Evaluate on the untouched 2025 test set
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not heavy rain", "Heavy rain"],
        zero_division=0,
    )
)

print(
    "Average precision:",
    round(average_precision_score(y_test, probabilities), 4),
)

# 7. Save the model and feature list
joblib.dump(model, MODEL_DIR / "fixed_run_heavy_rain_model.pkl")
joblib.dump(features, MODEL_DIR / "fixed_run_model_features.pkl")

print("\nModel saved in the models folder.")
