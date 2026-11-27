import pandas as pd
from pathlib import Path

folder = Path("data/chirps/expanded")

sample = pd.read_csv(folder / "expanded_sample_dates.csv")

sample["date"] = pd.to_datetime(sample["date"]).dt.strftime("%Y-%m-%d")
sample["location"] = sample["location"].astype(str)

expected = set(zip(sample["date"], sample["location"]))

for product in ["rnl", "sat"]:
    print("\n" + "=" * 50)
    print(f"{product.upper()} VALIDATION CHECK")
    print("=" * 50)

    results_file = folder / f"expanded_results_{product}.csv"
    failures_file = folder / f"expanded_failures_{product}.csv"

    results = pd.read_csv(results_file)
    failures = pd.read_csv(failures_file)

    for df in [results, failures]:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["location"] = df["location"].astype(str)

    successful = set(zip(results["date"], results["location"]))
    failed = set(zip(failures["date"], failures["location"]))

    missing = expected - successful - failed
    successful = expected & successful
    failed = expected & failed

    print(f"Expected records: {len(expected)}")
    print(f"Successful records: {len(successful)}")
    print(f"Failed records: {len(failed)}")
    print(f"Unaccounted records: {len(missing)}")
    print(f"Duplicate result rows: {results.duplicated(['date', 'location']).sum()}")

    if missing:
        print("\nFirst missing records:")
        for item in sorted(missing)[:10]:
            print(item)