
import pandas as pd

# Load the predictions made by our model
df = pd.read_csv("data/rainfall_predictions.csv")

# Calculate the absolute prediction error
df["absolute_error"] = (
    df["actual_rainfall_mm"] - df["predicted_rainfall_mm"]
).abs()

# Sort from largest error to smallest error
df = df.sort_values(
    by="absolute_error",
    ascending=False
)

# Display the 10 largest errors
print("\n--- TOP 10 LARGEST RAINFALL PREDICTION ERRORS ---")

print(
    df[
        [
            "date",
            "location",
            "actual_rainfall_mm",
            "predicted_rainfall_mm",
            "absolute_error",
        ]
    ].head(10).to_string(index=False)
)
