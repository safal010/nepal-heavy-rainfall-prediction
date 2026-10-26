
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_FILE = "data/fixed_run_forecast_training_data.csv"

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

# Evaluate only the later test period
test = df[df["date"] >= "2025-01-01"].copy()

actual = test["actual_rainfall_mm"]
forecast = test["forecast_rainfall_mm"]

mae = mean_absolute_error(actual, forecast)
rmse = np.sqrt(mean_squared_error(actual, forecast))

print("RAINFALL FORECAST EVALUATION")
print("-" * 40)
print("Test records:", len(test))
print("Mean actual rainfall:", round(actual.mean(), 2), "mm")
print("Mean forecast rainfall:", round(forecast.mean(), 2), "mm")
print("MAE:", round(mae, 2), "mm")
print("RMSE:", round(rmse, 2), "mm")

print("\nHEAVY-RAIN CASES")
heavy = test[test["actual_rainfall_mm"] >= 50]

print("Actual heavy-rain cases:", len(heavy))
print(
    "Forecast rainfall >= 50 mm:",
    int((heavy["forecast_rainfall_mm"] >= 50).sum())
)

print("\nLargest rainfall errors:")
test["absolute_error_mm"] = (actual - forecast).abs()

print(
    test.nlargest(10, "absolute_error_mm")[
        [
            "date",
            "location",
            "forecast_rainfall_mm",
            "actual_rainfall_mm",
            "absolute_error_mm",
        ]
    ].to_string(index=False)
)
