from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score

DATA_FILE = Path("data/fixed_run_forecast_training_data.csv")
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_era5_daily(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "UTC",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    daily = resp.json()["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "era5_rain_mm": daily["precipitation_sum"],
    })


def main():
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    start_str = df["date"].min().strftime("%Y-%m-%d")
    end_str = df["date"].max().strftime("%Y-%m-%d")

    # Fetch ERA5 daily precipitation for each unique city (6 fast requests)
    era5_frames = []
    locs = df[["location", "latitude", "longitude"]].drop_duplicates()
    for _, row in locs.iterrows():
        print(f"Fetching ERA5 for {row['location']}...")
        sub = fetch_era5_daily(row["latitude"], row["longitude"], start_str, end_str)
        sub["location"] = row["location"]
        era5_frames.append(sub)
        time.sleep(0.3)

    era5 = pd.concat(era5_frames, ignore_index=True)
    merged = df.merge(era5, on=["date", "location"], how="left")
    merged = merged.sort_values(["location", "date"]).reset_index(drop=True)
    merged["era5_prev_day"] = merged.groupby("location")["era5_rain_mm"].shift(1)
    merged["era5_next_day"] = merged.groupby("location")["era5_rain_mm"].shift(-1)

    # 1. Inspect suspicious NASA POWER spikes (>= 50 mm actual, but ERA5 < 10 mm)
    suspects = merged[(merged["actual_rainfall_mm"] >= 50.0) & (merged["era5_rain_mm"] < 10.0)]
    print(f"\n--- SUSPECT NASA POWER SPIKES (actual >= 50 mm, ERA5 < 10 mm): {len(suspects)} rows ---")
    cols = [
        "date",
        "location",
        "forecast_rainfall_mm",
        "actual_rainfall_mm",
        "era5_rain_mm",
        "era5_prev_day",
        "era5_next_day",
    ]
    print(suspects[cols].sort_values("date").to_string(index=False))

    # 2. Check correlation across lag -1, 0, +1 (detects any date-shift bugs)
    print("\n--- DATE-ALIGNMENT CHECK (Correlation of Forecast vs Actual & ERA5) ---")
    for name, target_col in [("NASA POWER actual", "actual_rainfall_mm"), ("ERA5 reanalysis", "era5_rain_mm")]:
        c0 = merged["forecast_rainfall_mm"].corr(merged[target_col])
        cm1 = merged["forecast_rainfall_mm"].corr(merged.groupby("location")[target_col].shift(1))
        cp1 = merged["forecast_rainfall_mm"].corr(merged.groupby("location")[target_col].shift(-1))
        print(f"{name:20s} | lag -1: {cm1:.4f} | lag 0 (same day): {c0:.4f} | lag +1: {cp1:.4f}")

    # 3. Compare Raw ECMWF 2025 PR-AUC against NASA POWER vs ERA5 (>= 50 mm & >= 40 mm)
    test = merged[merged["date"] >= "2025-01-01"].copy()
    print("\n--- 2025 RAW ECMWF PR-AUC BY GROUND-TRUTH SOURCE ---")
    for label, y in [
        ("NASA POWER >= 50 mm (current)", (test["actual_rainfall_mm"] >= 50.0).astype(int)),
        ("NASA POWER >= 50 mm (excl. ERA5 < 10 glitches)", ((test["actual_rainfall_mm"] >= 50.0) & (test["era5_rain_mm"] >= 10.0)).astype(int)),
        ("ERA5 >= 50 mm", (test["era5_rain_mm"] >= 50.0).astype(int)),
        ("ERA5 >= 40 mm", (test["era5_rain_mm"] >= 40.0).astype(int)),
    ]:
        pr = average_precision_score(y, test["forecast_rainfall_mm"]) if y.sum() > 0 else float("nan")
        print(f"{label:48s} | events: {int(y.sum()):2d} | 2025 PR-AUC: {pr:.4f}")


if __name__ == "__main__":
    main()