
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

target = "heavy_rain_tomorrow"

for name, df in [("TRAINING DATA", train), ("TEST DATA", test)]:
    print(f"\n--- {name} ---")
    print(f"Total records: {len(df):,}")
    print(f"Heavy-rain days: {int(df[target].sum()):,}")
    print(f"Non-heavy-rain days: {int((df[target] == 0).sum()):,}")
    print(f"Heavy-rain percentage: {df[target].mean() * 100:.3f}%")

    print("\nHeavy-rain records by location:")
    print(
        df.groupby("location")[target]
        .agg(["sum", "count"])
        .to_string()
    )
