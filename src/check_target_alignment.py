
import pandas as pd

file_path = "data/nepal_weather.csv"
df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"]).dt.date

location = "Kathmandu"
target_dates = [
    pd.Timestamp("2025-09-02").date(),
    pd.Timestamp("2025-09-03").date(),
]

rainfall_column = "PRECTOTCORR"

print("--- ORIGINAL OBSERVATIONS ---")

for target_date in target_dates:
    rows = df[
        (df["location"] == location)
        & (df["date"] == target_date)
    ]

    print(f"\nDate: {target_date}")
    print("Number of records:", len(rows))

    if not rows.empty:
        print(rows[["date", "location", rainfall_column]].to_string(index=False))

print("\n--- RAINFALL SUMMARY FOR KATHMANDU ---")

kathmandu = df[df["location"] == location]

print(kathmandu[rainfall_column].describe())

print("\n--- RAINFALL ON NEARBY DATES ---")

nearby_dates = [
    pd.Timestamp("2025-08-31").date(),
    pd.Timestamp("2025-09-01").date(),
    pd.Timestamp("2025-09-02").date(),
    pd.Timestamp("2025-09-03").date(),
    pd.Timestamp("2025-09-04").date(),
    pd.Timestamp("2025-09-05").date(),
]

nearby = kathmandu[kathmandu["date"].isin(nearby_dates)]

print(
    nearby[["date", "location", rainfall_column]]
    .sort_values("date")
    .to_string(index=False)
)
