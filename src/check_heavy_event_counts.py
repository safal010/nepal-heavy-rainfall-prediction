
import pandas as pd

FILES = {
    "RNL": "data/chirps/expanded/expanded_results_rnl.csv",
    "SAT": "data/chirps/expanded/expanded_results_sat.csv"
}

for product, file_path in FILES.items():

    print("\n" + "=" * 55)
    print(f"{product} HEAVY RAINFALL COUNTS")
    print("=" * 55)

    df = pd.read_csv(file_path)

    # Extract year from date
    df["year"] = pd.to_datetime(df["date"]).dt.year

    # Count actual heavy rainfall events by year
    print("\nHeavy rainfall events by year:")

    yearly_counts = df.groupby("year")["heavy_nasa"].agg(
        heavy_events="sum",
        total_samples="count"
    )

    print(yearly_counts.to_string())

    # Count actual heavy rainfall events by location
    print("\nHeavy rainfall events by location:")

    location_counts = df.groupby("location")["heavy_nasa"].agg(
        heavy_events="sum",
        total_samples="count"
    )

    print(location_counts.to_string())
