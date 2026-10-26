
import pandas as pd

df = pd.read_csv("data/nepal_weather.csv")

rainfall = df["PRECTOTCORR"]

print("\n--- RAINFALL DATA SUMMARY ---")
print(rainfall.describe().round(2))

print("\n--- RAINFALL CATEGORY COUNTS ---")

categories = {
    "No rain (0 mm)": rainfall == 0,
    "Light rain (greater than 0 to 10 mm)": (
        (rainfall > 0) & (rainfall <= 10)
    ),
    "Moderate rain (greater than 10 to 50 mm)": (
        (rainfall > 10) & (rainfall < 50)
    ),
    "Heavy rain (50 mm or more)": rainfall >= 50,
}

for label, condition in categories.items():
    count = int(condition.sum())
    percentage = count / len(df) * 100

    print(f"{label}: {count:,} days ({percentage:.2f}%)")

print("\n--- HEAVY RAIN DAYS BY LOCATION ---")

heavy_by_location = (
    df.assign(heavy_rain=rainfall >= 50)
    .groupby("location")["heavy_rain"]
    .agg(["sum", "count"])
)

heavy_by_location["percentage"] = (
    heavy_by_location["sum"] / heavy_by_location["count"] * 100
)

print(heavy_by_location.round(2).to_string())
