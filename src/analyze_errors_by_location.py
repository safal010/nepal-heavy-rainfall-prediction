
import pandas as pd

# Load the model's predictions
df = pd.read_csv("data/rainfall_predictions.csv")

# Calculate absolute error for every prediction
df["absolute_error"] = (
    df["actual_rainfall_mm"] - df["predicted_rainfall_mm"]
).abs()

# Calculate prediction performance for each location
results = df.groupby("location").agg(
    number_of_days=("absolute_error", "count"),
    average_actual_rainfall=("actual_rainfall_mm", "mean"),
    average_predicted_rainfall=("predicted_rainfall_mm", "mean"),
    mean_absolute_error=("absolute_error", "mean"),
)

# Display locations with the largest errors first
results = results.sort_values(
    by="mean_absolute_error",
    ascending=False
)

print("\n--- PREDICTION ERRORS BY LOCATION ---")
print(results.round(2).to_string())
