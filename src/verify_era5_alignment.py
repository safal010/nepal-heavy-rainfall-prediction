from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path("data/fixed_run_forecast_training_data.csv")
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def fetch_era5_ktm(locs: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    frames = []
    for _, row in locs.iterrows():
        params = {
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "start_date": start,
            "end_date": end,
            "daily": "precipitation_sum",
            "models": "era5",
            "timezone": "Asia/Kathmandu",
        }
        resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
        resp.raise_for_status()
        daily = resp.json()["daily"]
        frames.append(
            pd.DataFrame({
                "date": pd.to_datetime(daily["time"]),
                "location": row["location"],
                "era5_mm": daily["precipitation_sum"],
            })
        )
        time.sleep(0.25)
    return pd.concat(frames, ignore_index=True)


def fit_ratios(tr: pd.DataFrame, prior_mm: float = 300.0) -> tuple[dict, float]:
    rainy = tr[tr["forecast_rainfall_mm"] >= 5.0]
    g = rainy["era5_mm"].sum() / rainy["forecast_rainfall_mm"].sum()
    ratios = {
        loc: (grp["era5_mm"].sum() + prior_mm * g) / (grp["forecast_rainfall_mm"].sum() + prior_mm)
        for loc, grp in rainy.groupby("location")
    }
    return ratios, g


def benchmark_era5(df: pd.DataFrame, thresh_mm: float = 50.0):
    data = df.sort_values(["location", "date"]).reset_index(drop=True)
    data["y"] = (data["era5_mm"] >= thresh_mm).astype(int)
    data["storm_week"] = data["date"].dt.strftime("%Y-%W")

    day_sum = data.groupby("date")["forecast_rainfall_mm"].transform("sum")
    day_cnt = data.groupby("date")["forecast_rainfall_mm"].transform("count")
    data["other_cities_mean_rain"] = (day_sum - data["forecast_rainfall_mm"]) / np.maximum(day_cnt - 1, 1)
    data["prev_day_forecast_rain"] = data.groupby("location")["forecast_rainfall_mm"].shift(1).fillna(0.0)

    train = data[data["date"] < "2025-01-01"].reset_index(drop=True)
    test = data[data["date"] >= "2025-01-01"].reset_index(drop=True)
    cv = StratifiedGroupKFold(n_splits=5)

    print(
        f"\n--- CLEAN ERA5 (Asia/Kathmandu) >= {thresh_mm:.0f} mm "
        f"(2024: {int(train['y'].sum())} events, 2025: {int(test['y'].sum())} events) ---"
    )

    def run(name, fn):
        oof = np.zeros(len(train))
        for tr_i, va_i in cv.split(train, train["y"], groups=train["storm_week"]):
            oof[va_i] = fn(train.iloc[tr_i], train.iloc[va_i])
        preds = fn(train, test)
        brier = (
            f" | 2025 Brier: {brier_score_loss(test['y'], preds):.4f}"
            if 0.0 <= preds.min() and preds.max() <= 1.0
            else ""
        )
        print(
            f"  {name:34s} | 2024 CV: {average_precision_score(train['y'], oof):.4f} "
            f"| 2025 PR-AUC: {average_precision_score(test['y'], preds):.4f}{brier}"
        )

    run("0. Raw ECMWF forecast_mm", lambda tr, va: va["forecast_rainfall_mm"].values)

    def bias_only(tr, va):
        ratios, g = fit_ratios(tr)
        tr_c = (tr["forecast_rainfall_mm"] * tr["location"].map(ratios).fillna(g)).values.reshape(-1, 1)
        va_c = (va["forecast_rainfall_mm"] * va["location"].map(ratios).fillna(g)).values.reshape(-1, 1)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
        clf.fit(tr_c, tr["y"])
        return clf.predict_proba(va_c)[:, 1]

    run("1. Shrunk Loc-Bias -> LogReg", bias_only)

    def bias_reg_persist(tr, va):
        ratios, g = fit_ratios(tr)
        X_tr = np.column_stack([
            tr["forecast_rainfall_mm"] * tr["location"].map(ratios).fillna(g),
            tr["other_cities_mean_rain"],
            tr["prev_day_forecast_rain"],
        ])
        X_va = np.column_stack([
            va["forecast_rainfall_mm"] * va["location"].map(ratios).fillna(g),
            va["other_cities_mean_rain"],
            va["prev_day_forecast_rain"],
        ])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5))
        clf.fit(X_tr, tr["y"])
        return clf.predict_proba(X_va)[:, 1]

    run("2. Loc-Bias + Regional + Prev-Day", bias_reg_persist)

    # Test with extra meteorology columns if present in CSV
    extra_cols = [
        c for c in ["humidity_mean_pct", "temperature_mean_c", "wind_speed_mean_kmh"]
        if c in data.columns
    ]

    def hgb_clean(tr, va):
        ratios, g = fit_ratios(tr)
        def make_X(d):
            cols = [
                d["forecast_rainfall_mm"] * d["location"].map(ratios).fillna(g),
                d["forecast_rainfall_mm"],
                d["other_cities_mean_rain"],
                d["prev_day_forecast_rain"],
            ] + [d[c] for c in extra_cols]
            return np.column_stack(cols)

        clf = HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=20, l2_regularization=5.0,
            learning_rate=0.05, max_iter=150, random_state=42,
        )
        clf.fit(make_X(tr), tr["y"])
        return clf.predict_proba(make_X(va))[:, 1]

    run("3. Regularized HGB (Bias+Met)", hgb_clean)


def probe_previous_runs_api():
    print("\n--- MULTI-MODEL 2024 COVERAGE PROBE (Pokhara, Aug 1-5 2024) ---")
    models = ["ecmwf_ifs025", "ecmwf_ifs", "gfs_seamless", "icon_seamless", "gem_seamless"]
    vars_to_check = ["precipitation_previous_day1", "cape_previous_day1", "dew_point_2m_previous_day1"]
    for m in models:
        params = {
            "latitude": 28.2096,
            "longitude": 83.9856,
            "start_date": "2024-08-01",
            "end_date": "2024-08-05",
            "hourly": ",".join(vars_to_check),
            "models": m,
            "timezone": "Asia/Kathmandu",
        }
        try:
            r = requests.get(PREV_RUNS_URL, params=params, timeout=20)
            if r.status_code != 200:
                print(f"  {m:16s} | HTTP {r.status_code}: {r.text[:90]}")
                continue
            hourly = r.json().get("hourly", {})
            counts = {
                v.replace("_previous_day1", ""): sum(x is not None for x in hourly.get(v, []))
                for v in vars_to_check
            }
            print(f"  {m:16s} | non-null hours (out of 120): {counts}")
        except Exception as e:
            print(f"  {m:16s} | ERROR: {e}")
        time.sleep(0.2)


def main():
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    locs = df[["location", "latitude", "longitude"]].drop_duplicates()
    start, end = df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")

    era5 = fetch_era5_ktm(locs, start, end)
    merged = df.merge(era5, on=["date", "location"], how="left")

    print("--- 2024 PER-LOCATION BIAS VS CLEAN ERA5 (Asia/Kathmandu, ERA5 >= 5 mm) ---")
    tr_rainy = merged[(merged["date"] < "2025-01-01") & (merged["era5_mm"] >= 5.0)]
    summary = (
        tr_rainy.groupby("location")
        .agg(
            rainy_days=("era5_mm", "count"),
            heavy_50=("era5_mm", lambda s: int((s >= 50.0).sum())),
            mean_ecmwf=("forecast_rainfall_mm", "mean"),
            mean_era5=("era5_mm", "mean"),
        )
        .assign(ratio=lambda d: d["mean_era5"] / d["mean_ecmwf"])
    )
    print(summary.round(2).to_string())

    benchmark_era5(merged, thresh_mm=50.0)
    benchmark_era5(merged, thresh_mm=40.0)
    probe_previous_runs_api()


if __name__ == "__main__":
    main()