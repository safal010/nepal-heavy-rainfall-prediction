
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Load the same test data used to evaluate the model
test = pd.read_csv("data/test.csv")
model = joblib.load("models/rainfall_model.pkl")
features = joblib.load("models/model_features.pkl")

target = "rainfall_tomorrow_mm"

# Get the actual rainfall amounts
y_actual = test[target]

# Baseline: predict the average rainfall from training data
train = pd.read_csv("data/train.csv")
average_rainfall = train[target].mean()
baseline_predictions = [average_rainfall] * len(test)

# Random Forest: predict using weather features
X_test = test[features]
rf_predictions = model.predict(X_test)

# Calculate both sets of errors
print("\n--- BASELINE ---")
print(f"Average rainfall: {average_rainfall:.2f} mm")
print(f"MAE: {mean_absolute_error(y_actual, baseline_predictions):.2f} mm")
print(f"RMSE: {root_mean_squared_error(y_actual, baseline_predictions):.2f} mm")

print("\n--- RANDOM FOREST ---")
print(f"MAE: {mean_absolute_error(y_actual, rf_predictions):.2f} mm")
print(f"RMSE: {root_mean_squared_error(y_actual, rf_predictions):.2f} mm")
