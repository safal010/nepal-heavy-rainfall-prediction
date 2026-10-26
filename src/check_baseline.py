
import pandas as pd
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/forecast_comparison_sample.csv")

actual = df["observed_mm"]
forecast = df["forecast_mm"]

# Simple baseline: always predict zero rainfall
zero_predictions = [0.0] * len(df)

forecast_mae = mean_absolute_error(actual, forecast)
baseline_mae = mean_absolute_error(actual, zero_predictions)

print("Number of comparisons:", len(df))
print("Forecast MAE:", round(forecast_mae, 2), "mm")
print("Zero-rainfall baseline MAE:", round(baseline_mae, 2), "mm")

if forecast_mae < baseline_mae:
    print("Result: Forecast beats the zero-rainfall baseline.")
else:
    print("Result: Forecast does not beat the zero-rainfall baseline.")
