from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests

BASE_CSV = Path("data/fixed_run_forecast_training_data.csv")
OUT_CSV = Path("data/multimodel_nepal_rain.csv")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
TZ = "Asia/Kathmandu"

MODELS = {
    "ecmwf": ("ecmwf_ifs025", ["precipitation_previous_day1", "dew_point_2m_previous_day1"]),
    "gfs": ("gfs_seamless", ["precipitation_previous_day1", "dew_point_2m_previous_day1", "cape_previous_day1"]),
    "icon": ("icon_seamless", ["precipitation_previous_day1", "dew_point_2m_previous_day1"]),
    "gem": ("gem_seamless", ["precipitation_previous_day1", "dew_point_2m_previous_day1"]),
}


def get_with_retry(url: str, params: dict, max_retries: int = 4) -> dict:
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=45)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 2.0 * (attempt + 1)
            print(f"    HTTP {resp.status_code}, retrying in {wait:.0f}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed after {max_retries} retries: {url} {params}")


def year_chunks(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> list[tuple[str, str]]:
    chunks = []
    for yr in range(start_dt.year, end_dt.year + 1):
        s = max(start_dt, pd.Timestamp(f"{yr}-01-01"))
        e = min(end_dt, pd.Timestamp(f"{yr}-12-31"))
        chunks.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    return chunks


def fetch_era5(locs: pd.DataFrame, start_str: str, end_str: str) -> pd.DataFrame:
    frames = []
    for _, row in locs.iterrows():
        data = get_with_retry(
            ARCHIVE_URL,
            {
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "start_date": start_str,
                "end_date": end_str,
                "daily": "precipitation_sum",
                "models": "era5",
                "timezone": TZ,
            },
        )
        daily = data["daily"]
        frames.append(
            pd.DataFrame({
                "date": pd.to_datetime(daily["time"]),
                "location": row["location"],
                "actual_era5_mm": pd.to_numeric(daily["precipitation_sum"], errors="coerce"),
            })
        )
        time.sleep(0.25)
    return pd.concat(frames, ignore_index=True)


def fetch_model_city(
    short_name: str,
    api_model: str,
    hourly_vars: list[str],
    lat: float,
    lon: float,
    location: str,
    chunks: list[tuple[str, str]],
) -> pd.DataFrame:
    hourly_parts = []
    for s_str, e_str in chunks:
        data = get_with_retry(
            PREV_RUNS_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": s_str,
                "end_date": e_str,
                "hourly": ",".join(hourly_vars),
                "models": api_model,
                "timezone": TZ,
            },
        )
        h = data["hourly"]
        part = pd.DataFrame({"time": pd.to_datetime(h["time"])})
        part["precip"] = pd.to_numeric(h["precipitation_previous_day1"], errors="coerce")
        part["dew"] = pd.to_numeric(h["dew_point_2m_previous_day1"], errors="coerce")
        if "cape_previous_day1" in hourly_vars:
            part["cape"] = pd.to_numeric(h["cape_previous_day1"], errors="coerce")
        hourly_parts.append(part)
        time.sleep(0.35)

    hourly = (
        pd.concat(hourly_parts, ignore_index=True)
        .drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    hourly["precip_3h"] = hourly["precip"].rolling(3, min_periods=3).sum()
    hourly["date"] = hourly["time"].dt.floor("D")

    grp = hourly.groupby("date")
    out = pd.DataFrame({
        "date": grp.size().index,
        "location": location,
        f"{short_name}_rain_mm": grp["precip"].sum(min_count=18).values,
        f"{short_name}_max_1h_mm": grp["precip"].max().values,
        f"{short_name}_max_3h_mm": grp["precip_3h"].max().values,
        f"{short_name}_dew_mean_c": grp["dew"].mean().values,
    })
    if "cape" in hourly.columns:
        out[f"{short_name}_cape_max"] = grp["cape"].max().values
    return out


def main():
    base = pd.read_csv(BASE_CSV)
    base["date"] = pd.to_datetime(base["date"])
    locs = base[["location", "latitude", "longitude"]].drop_duplicates().sort_values("location")

    start_dt, end_dt = base["date"].min(), base["date"].max()
    start_str, end_str = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    chunks = year_chunks(start_dt, end_dt)
    print(f"Building multi-model dataset for {len(locs)} cities: {start_str} -> {end_str}")

    df = base[["date", "location", "latitude", "longitude", "forecast_rainfall_mm"]].rename(
        columns={"forecast_rainfall_mm": "orig_ecmwf_rain_mm"}
    )

    print("1/5 Fetching clean ERA5 (Asia/Kathmandu) ground truth...")
    era5 = fetch_era5(locs, start_str, end_str)
    df = df.merge(era5, on=["date", "location"], how="left")

    for idx, (short_name, (api_model, hourly_vars)) in enumerate(MODELS.items(), start=2):
        print(f"{idx}/5 Fetching {short_name.upper()} ({api_model})...")
        city_frames = []
        for _, row in locs.iterrows():
            sub = fetch_model_city(
                short_name, api_model, hourly_vars,
                row["latitude"], row["longitude"], row["location"], chunks
            )
            city_frames.append(sub)
        model_df = pd.concat(city_frames, ignore_index=True)
        df = df.merge(model_df, on=["date", "location"], how="left")

    df = df.sort_values(["date", "location"]).reset_index(drop=True)
    df["heavy_rain_50mm"] = (df["actual_era5_mm"] >= 50.0).astype(int)
    df["heavy_rain_40mm"] = (df["actual_era5_mm"] >= 40.0).astype(int)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows x {df.shape[1]} cols -> {OUT_CSV}\n")

    print("--- DATA COVERAGE & CORRELATION WITH ERA5 (Asia/Kathmandu) ---")
    for year in [2024, 2025]:
        sub = df[df["date"].dt.year == year]
        print(f"\nYear {year} (rows={len(sub)}, >=50mm events={int(sub['heavy_rain_50mm'].sum())}):")
        for col in ["orig_ecmwf_rain_mm", "ecmwf_rain_mm", "gfs_rain_mm", "icon_rain_mm", "gem_rain_mm"]:
            valid = sub[col].notna().sum()
            corr = sub[col].corr(sub["actual_era5_mm"])
            print(f"  {col:20s} | non-null: {valid:4d}/{len(sub)} | corr vs ERA5: {corr:.4f}")


if __name__ == "__main__":
    main()