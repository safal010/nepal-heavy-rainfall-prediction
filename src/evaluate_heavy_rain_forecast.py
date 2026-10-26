
import pandas as pd
from pathlib import Path

# ==========================================
# 1. LOAD FORECAST COMPARISON DATA
# ==========================================

file_path = Path("data/forecast_comparison_year.csv")

if not file_path.exists():
    raise FileNotFoundError(
        f"File not found: {file_path}. "
        "Run src/evaluate_year_round.py first."
    )

df = pd.read_csv(file_path)

required_columns = {
    "date",
    "location",
    "forecast_mm",
    "observed_mm",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Remove rows without valid rainfall values.
df["forecast_mm"] = pd.to_numeric(
    df["forecast_mm"], errors="coerce"
)
df["observed_mm"] = pd.to_numeric(
    df["observed_mm"], errors="coerce"
)

df = df.dropna(
    subset=["forecast_mm", "observed_mm"]
).copy()

if df.empty:
    raise ValueError("No valid forecast comparisons found.")

# ==========================================
# 2. DEFINE HEAVY-RAINFALL THRESHOLD
# ==========================================

# For this project, heavy rainfall means
# at least 50 mm in one day.
threshold = 50.0

df["actual_heavy"] = (
    df["observed_mm"] >= threshold
).astype(int)

df["predicted_heavy"] = (
    df["forecast_mm"] >= threshold
).astype(int)

# ==========================================
# 3. CALCULATE CONFUSION MATRIX COUNTS
# ==========================================

actual = df["actual_heavy"]
predicted = df["predicted_heavy"]

true_positive = int(
    ((actual == 1) & (predicted == 1)).sum()
)

false_positive = int(
    ((actual == 0) & (predicted == 1)).sum()
)

false_negative = int(
    ((actual == 1) & (predicted == 0)).sum()
)

true_negative = int(
    ((actual == 0) & (predicted == 0)).sum()
)

# ==========================================
# 4. CALCULATE EVALUATION METRICS
# ==========================================

total = len(df)

accuracy = (
    (true_positive + true_negative) / total
    if total else 0.0
)

precision = (
    true_positive / (true_positive + false_positive)
    if (true_positive + false_positive) else 0.0
)

recall = (
    true_positive / (true_positive + false_negative)
    if (true_positive + false_negative) else 0.0
)

f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) else 0.0
)

# ==========================================
# 5. DISPLAY RESULTS
# ==========================================

print("\n======================================")
print("HEAVY-RAINFALL FORECAST EVALUATION")
print("======================================")

print(f"Heavy-rainfall threshold: {threshold:.0f} mm/day")
print(f"Total valid comparisons: {total}")
print(f"Actual heavy-rainfall cases: {int(actual.sum())}")
print(f"Predicted heavy-rainfall cases: {int(predicted.sum())}")

print("\n--- CONFUSION MATRIX COUNTS ---")
print(f"True positives  (correct warnings): {true_positive}")
print(f"False positives (false alarms):     {false_positive}")
print(f"False negatives (missed events):    {false_negative}")
print(f"True negatives  (correct no-alert): {true_negative}")

print("\n--- METRICS ---")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1_score:.3f}")

# ==========================================
# 6. DISPLAY HEAVY-RAINFALL CASES
# ==========================================

print("\n--- HEAVY-RAINFALL CASES ---")

cases = df[
    (df["actual_heavy"] == 1)
    | (df["predicted_heavy"] == 1)
]

if cases.empty:
    print("No actual or predicted heavy-rainfall cases found.")
else:
    print(
        cases[
            [
                "date",
                "location",
                "forecast_mm",
                "observed_mm",
                "actual_heavy",
                "predicted_heavy",
            ]
        ].to_string(index=False)
    )

# ==========================================
# 7. SAVE EVALUATION RESULTS
# ==========================================

output_file = Path("data/heavy_rain_forecast_evaluation.csv")

df.to_csv(output_file, index=False)

print(f"\nDetailed results saved to: {output_file}")

print("\nNote: This test uses only selected dates.")
print("We need more dates and seasons to evaluate")
print("the reliability of heavy-rainfall warnings.")
