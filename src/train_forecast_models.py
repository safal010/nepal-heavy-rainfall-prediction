
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.dummy import DummyRegressor, DummyClassifier


# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "forecast_training_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load the dataset
print("Loading forecast dataset...")

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

# New feature: extract month from the date
df["month"] = df["date"].dt.month


# 3. Select input features and targets
feature_columns = [
    "latitude",
    "longitude",
    "month",
    "temperature_mean_c",
    "humidity_mean_pct",
    "forecast_rainfall_mm",
    "wind_speed_mean_kmh",
]

X = df[feature_columns]
y_rainfall = df["actual_rainfall_mm"]
y_heavy = df["heavy_rain_tomorrow"]


# 4. Split chronologically: 60% train, 20% validation, 20% test
unique_dates = sorted(df["date"].unique())

train_end_index = int(len(unique_dates) * 0.60)
validation_end_index = int(len(unique_dates) * 0.80)

validation_start_date = unique_dates[train_end_index]
test_start_date = unique_dates[validation_end_index]

train_mask = df["date"] < validation_start_date

validation_mask = (
    (df["date"] >= validation_start_date)
    & (df["date"] < test_start_date)
)

test_mask = df["date"] >= test_start_date

X_train = X.loc[train_mask]
X_validation = X.loc[validation_mask]
X_test = X.loc[test_mask]

y_rain_train = y_rainfall.loc[train_mask]
y_rain_validation = y_rainfall.loc[validation_mask]
y_rain_test = y_rainfall.loc[test_mask]

y_heavy_train = y_heavy.loc[train_mask]
y_heavy_validation = y_heavy.loc[validation_mask]
y_heavy_test = y_heavy.loc[test_mask]


print("\n--- CHRONOLOGICAL DATA SPLIT ---")
print("Training ends before:", validation_start_date)
print("Validation starts:", validation_start_date)
print("Final testing starts:", test_start_date)

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_validation))
print("Testing rows:", len(X_test))

print("Heavy-rain cases in training:", int(y_heavy_train.sum()))
print("Heavy-rain cases in validation:", int(y_heavy_validation.sum()))
print("Heavy-rain cases in testing:", int(y_heavy_test.sum()))

if (
    y_heavy_train.sum() == 0
    or y_heavy_validation.sum() == 0
    or y_heavy_test.sum() == 0
):
    raise ValueError(
        "A data split contains no heavy-rain cases. "
        "Inspect the date ranges before continuing."
    )


# 5. Train and evaluate the rainfall amount model
print("\n--- TRAINING RAINFALL REGRESSION MODEL ---")

rain_model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

rain_model.fit(X_train, y_rain_train)

rain_predictions = rain_model.predict(X_test)

rain_mae = mean_absolute_error(
    y_rain_test, rain_predictions
)

rain_rmse = mean_squared_error(
    y_rain_test, rain_predictions
) ** 0.5


# Baseline: always predict average training rainfall
rain_baseline = DummyRegressor(strategy="mean")
rain_baseline.fit(X_train, y_rain_train)

baseline_predictions = rain_baseline.predict(X_test)

baseline_mae = mean_absolute_error(
    y_rain_test, baseline_predictions
)

baseline_rmse = mean_squared_error(
    y_rain_test, baseline_predictions
) ** 0.5


print("\n--- RAINFALL AMOUNT TEST RESULTS ---")
print(f"Random Forest MAE:  {rain_mae:.3f} mm")
print(f"Random Forest RMSE: {rain_rmse:.3f} mm")
print(f"Baseline MAE:       {baseline_mae:.3f} mm")
print(f"Baseline RMSE:      {baseline_rmse:.3f} mm")


# 6. Train the heavy-rain classification model
print("\n--- TRAINING HEAVY-RAIN CLASSIFIER ---")

heavy_model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

heavy_model.fit(X_train, y_heavy_train)


# 7. Evaluate candidate thresholds on validation data only
validation_probabilities = heavy_model.predict_proba(
    X_validation
)[:, 1]

print("\n--- VALIDATION THRESHOLD COMPARISON ---")
print("Validation records:", len(X_validation))
print("Actual heavy-rain cases:", int(y_heavy_validation.sum()))

for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:

    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_heavy_validation,
        validation_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_heavy_validation,
        validation_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_heavy_validation,
        validation_predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_heavy_validation,
        validation_predictions,
        labels=[0, 1],
    ).ravel()

    print(f"\nThreshold: {threshold:.2f}")
    print(
        "Predicted heavy-rain cases:",
        int(validation_predictions.sum()),
    )
    print(f"True positives: {tp}")
    print(f"False alarms: {fp}")
    print(f"Missed heavy-rain cases: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")


# Validation baseline
heavy_baseline = DummyClassifier(strategy="most_frequent")
heavy_baseline.fit(X_train, y_heavy_train)

baseline_validation_predictions = heavy_baseline.predict(
    X_validation
)

print("\n--- VALIDATION BASELINE ---")
print(
    "Baseline recall:",
    f"{recall_score(y_heavy_validation, baseline_validation_predictions, zero_division=0):.3f}",
)
print(
    "Baseline accuracy:",
    f"{(baseline_validation_predictions == y_heavy_validation).mean():.3f}",
)


# 8. Save trained models and feature names
joblib.dump(
    rain_model,
    MODEL_DIR / "forecast_rainfall_model.pkl",
)

joblib.dump(
    heavy_model,
    MODEL_DIR / "forecast_heavy_rain_model.pkl",
)

joblib.dump(
    feature_columns,
    MODEL_DIR / "forecast_model_features.pkl",
)

print("\nModels saved in:", MODEL_DIR)
print("Training and validation completed.")
print("The final test set has not been used to select a threshold.")
