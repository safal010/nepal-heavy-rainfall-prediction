
import pandas as pd

# Load NASA POWER observed rainfall data
df = pd.read_csv("data/nepal_weather.csv")
df["date"] = pd.to_datetime(df["date"])

# Sort dates within each location
df = df.sort_values(["location", "date"]).reset_index(drop=True)

# Get rainfall for the next day and the day after that
df["next_day_rainfall"] = (
    df.groupby("location")["PRECTOTCORR"].shift(-1)
)

df["day_after_next_rainfall"] = (
    df.groupby("location")["PRECTOTCORR"].shift(-2)
)

# Heavy-rain threshold
HEAVY_RAIN_THRESHOLD_MM = 50.0

# Dates to investigate
checks = [
    ("Kathmandu", "2025-02-27", "2025-03-02"),
    ("Nepalgunj", "2025-06-09", "2025-06-12"),
    ("Pokhara", "2024-09-25", "2024-09-28"),
]

for location, start, end in checks:
    print(f"\n{'=' * 65}")
    print(f"{location}: {start} to {end}")
    print("=" * 65)

    rows = df[
        (df["location"] == location)
        & (df["date"] >= start)
        & (df["date"] <= end)
    ].copy()

    if rows.empty:
        print("No records found for this location and date range.")
        continue

    # Flag heavy rainfall on each displayed date
    rows["heavy_rain_today"] = (
        rows["PRECTOTCORR"] >= HEAVY_RAIN_THRESHOLD_MM
    )

    rows["heavy_rain_next_day"] = (
        rows["next_day_rainfall"] >= HEAVY_RAIN_THRESHOLD_MM
    )

    # Round values for easier reading
    columns = [
        "date",
        "PRECTOTCORR",
        "next_day_rainfall",
        "day_after_next_rainfall",
        "heavy_rain_today",
        "heavy_rain_next_day",
    ]

    print(rows[columns].to_string(index=False))

# Summarize extreme rainfall observations across the dataset
extreme = df[df["PRECTOTCORR"] >= HEAVY_RAIN_THRESHOLD_MM]

print(f"\n{'=' * 65}")
print("OVERALL EXTREME RAINFALL SUMMARY")
print("=" * 65)
print(f"Total records: {len(df)}")
print(f"Records with rainfall >= 50 mm: {len(extreme)}")

if not extreme.empty:
    print("\nLocations with the most extreme rainfall records:")
    print(
        extreme.groupby("location")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nTen highest rainfall observations:")
    print(
        extreme.sort_values("PRECTOTCORR", ascending=False)
        [["date", "location", "PRECTOTCORR"]]
        .head(10)
        .to_string(index=False)
    )
