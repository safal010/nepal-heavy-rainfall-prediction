
import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
)

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Load training and testing datasets
# --------------------------------------------------

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# --------------------------------------------------
# 2. Separate targets and model input features
# --------------------------------------------------

target_columns = [
    "rainfall_tomorrow_mm",
    "heavy_rain_tomorrow",
]

# Metadata is kept in CSV files for analysis.
# It must not be passed directly into the models.
metadata_columns = ["date", "location"]

X_train = train.drop(
    columns=target_columns + metadata_columns,
)

X_test = test.drop(
    columns=target_columns + metadata_columns,
)

y_train_rainfall = train["rainfall_tomorrow_mm"]
y_test_rainfall = test["rainfall_tomorrow_mm"]

y_train_heavy = train["heavy_rain_tomorrow"]
y_test_heavy = test["heavy_rain_tomorrow"]

# Ensure the training and testing inputs match
if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Training and testing feature columns do not match."
    )

print("\n--- DATASET INFORMATION ---")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Model input features:", len(X_train.columns))

print("\nFeatures being used:")
print(X_train.columns.tolist())

# --------------------------------------------------
# 3. Train rainfall-amount regression model
# --------------------------------------------------

print("\nTraining rainfall regression model...")

rainfall_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

rainfall_model.fit(
    X_train,
    y_train_rainfall,
)

rainfall_predictions = rainfall_model.predict(X_test)

mae = mean_absolute_error(
    y_test_rainfall,
    rainfall_predictions,
)

rmse = root_mean_squared_error(
    y_test_rainfall,
    rainfall_predictions,
)

print("\n--- RAINFALL AMOUNT RESULTS ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} mm")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mm")

# --------------------------------------------------
# 4. Train heavy-rain classification model
# --------------------------------------------------

print("\nTraining heavy-rain classification model...")

heavy_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

heavy_model.fit(
    X_train,
    y_train_heavy,
)

# Evaluate the default classification threshold
heavy_predictions = heavy_model.predict(X_test)

print("\n--- HEAVY RAIN RESULTS (DEFAULT THRESHOLD) ---")

print(
    "Precision:",
    round(
        precision_score(
            y_test_heavy,
            heavy_predictions,
            zero_division=0,
        ),
        3,
    ),
)

print(
    "Recall:",
    round(
        recall_score(
            y_test_heavy,
            heavy_predictions,
            zero_division=0,
        ),
        3,
    ),
)

print(
    "F1 score:",
    round(
        f1_score(
            y_test_heavy,
            heavy_predictions,
            zero_division=0,
        ),
        3,
    ),
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test_heavy, heavy_predictions))

# --------------------------------------------------
# 5. Save models and feature names
# --------------------------------------------------

Path("models").mkdir(exist_ok=True)

joblib.dump(
    rainfall_model,
    "models/rainfall_model.pkl",
)

joblib.dump(
    heavy_model,
    "models/heavy_rain_model.pkl",
)

# Save the exact feature order used during training.
joblib.dump(
    X_train.columns.tolist(),
    "models/model_features.pkl",
)

print("\nModels saved successfully in the models folder.")
