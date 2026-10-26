
import pandas as pd

# Load model predictions
df = pd.read_csv("data/rainfall_predictions.csv")

# Calculate absolute error
df["absolute_error"] = (
    df["actual_rainfall_mm"] - df["predicted_rainfall_mm"]
).abs()

# Group actual rainfall into categories
df["rainfall_category"] = pd.cut(
    df["actual_rainfall_mm"],
    bins=[-0.01, 0, 10, 50, float("inf")],
    labels=[
        "No rain",
        "Light rain (0-10 mm)",
        "Moderate rain (10-50 mm)",
        "Heavy rain (50+ mm)"
    ]
)

# Calculate error statistics for each category
results = df.groupby(
    "rainfall_category",
    observed=False
).agg(
    number_of_days=("absolute_error", "count"),
    average_actual_rainfall=("actual_rainfall_mm", "mean"),
    average_predicted_rainfall=("predicted_rainfall_mm", "mean"),
    mean_absolute_error=("absolute_error", "mean")
)

print("\n--- PREDICTION ERRORS BY RAINFALL CATEGORY ---")
print(results.round(2).to_string())
