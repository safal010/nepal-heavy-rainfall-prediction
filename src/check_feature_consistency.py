
import pandas as pd
import joblib

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

features = joblib.load("models/model_features.pkl")

print("\n--- FEATURE CONSISTENCY CHECK ---")

print("Number of features:", len(features))
print("All features in training:", all(f in train.columns for f in features))
print("All features in testing:", all(f in test.columns for f in features))

print("\n--- TARGET AND DATE INFORMATION ---")

for name, df in [("Training", train), ("Testing", test)]:
    print(f"\n{name}:")
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Rows:", len(df))
    print("Heavy-rain cases:", int(df["heavy_rain_tomorrow"].sum()))

print("\n--- POSSIBLE TARGET LEAKAGE CHECK ---")

suspicious = [
    f for f in features
    if "tomorrow" in f.lower()
    or f.lower() in ["rainfall_tomorrow_mm", "heavy_rain_tomorrow"]
]

print("Features that look like target leakage:", suspicious)
