
import pandas as pd

# Load the model's predictions
df = pd.read_csv("data/rainfall_predictions.csv")

# Calculate prediction error
df["absolute_error"] = (
    df["actual_rainfall_mm"] - df["predicted_rainfall_mm"]
).abs()

# Keep only heavy-rainfall cases
heavy = df[df["actual_rainfall_mm"] >= 50].copy()

print("\n--- HEAVY-RAINFALL ERRORS BY LOCATION ---")

if heavy.empty:
    print("No heavy-rainfall cases found.")
else:
    results = heavy.groupby("location").agg(
        heavy_rain_days=("absolute_error", "count"),
        average_actual_rainfall=("actual_rainfall_mm", "mean"),
        average_predicted_rainfall=("predicted_rainfall_mm", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        largest_error=("absolute_error", "max"),
    )

    results = results.sort_values(
        by="mean_absolute_error",
        ascending=False,
    )

    print(results.round(2).to_string())
