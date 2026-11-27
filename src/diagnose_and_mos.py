from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path("data/fixed_run_forecast_training_data.csv")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["location", "date"]).reset_index(drop=True)
    df["storm_week"] = df["date"].dt.strftime("%Y-%W")

    # Leak-free same-day regional signal (mean/max across the OTHER 5 cities)
    day_sum = df.groupby("date")["forecast_rainfall_mm"].transform("sum")
    day_cnt = df.groupby("date")["forecast_rainfall_mm"].transform("count")
    df["other_cities_mean_rain"] = (day_sum - df["forecast_rainfall_mm"]) / np.maximum(
        day_cnt - 1, 1
    )

    # Leak-free previous-day forecast (persistence)
    df["prev_day_forecast_rain"] = (
        df.groupby("location")["forecast_rainfall_mm"].shift(1).fillna(0.0)
    )
    return df


def inspect_2025_events(test: pd.DataFrame):
    print("--- ALL 18 ACTUAL HEAVY-RAIN EVENTS IN 2025 ---")
    events = test[test["heavy_rain_tomorrow"] == 1][
        [
            "date",
            "location",
            "forecast_rainfall_mm",
            "prev_day_forecast_rain",
            "other_cities_mean_rain",
            "actual_rainfall_mm",
        ]
    ].sort_values("actual_rainfall_mm", ascending=False)
    print(events.to_string(index=False))

    print("\n--- TOP 10 BIGGEST 2025 FALSE ALARMS (Highest ECMWF, actual < 50 mm) ---")
    false_alarms = (
        test[test["heavy_rain_tomorrow"] == 0]
        .sort_values("forecast_rainfall_mm", ascending=False)
        .head(10)[
            [
                "date",
                "location",
                "forecast_rainfall_mm",
                "other_cities_mean_rain",
                "actual_rainfall_mm",
            ]
        ]
    )
    print(false_alarms.to_string(index=False))
    print()


def fit_location_ratios(tr: pd.DataFrame, prior_mm: float = 300.0) -> tuple[dict, float]:
    """Empirical-Bayes shrunk per-location actual/forecast ratio on rainy days."""
    rainy = tr[tr["forecast_rainfall_mm"] >= 5.0]
    global_ratio = rainy["actual_rainfall_mm"].sum() / rainy["forecast_rainfall_mm"].sum()
    ratios = {}
    for loc, grp in rainy.groupby("location"):
        num = grp["actual_rainfall_mm"].sum() + prior_mm * global_ratio
        den = grp["forecast_rainfall_mm"].sum() + prior_mm
        ratios[loc] = num / den
    return ratios, global_ratio


def evaluate(name, train, test, fit_predict_fn):
    cv = StratifiedGroupKFold(n_splits=5)
    y_tr = train["heavy_rain_tomorrow"].values
    y_te = test["heavy_rain_tomorrow"].values

    oof = np.zeros(len(train))
    for tr_i, va_i in cv.split(train, y_tr, groups=train["storm_week"]):
        oof[va_i] = fit_predict_fn(train.iloc[tr_i], train.iloc[va_i])

    preds = fit_predict_fn(train, test)
    oof_pr = average_precision_score(y_tr, oof)
    te_pr = average_precision_score(y_te, preds)
    brier = (
        f" | 2025 Brier: {brier_score_loss(y_te, preds):.4f}"
        if 0.0 <= preds.min() and preds.max() <= 1.0
        else ""
    )
    print(f"{name:38s} | 2024 CV PR-AUC: {oof_pr:.4f} | 2025 PR-AUC: {te_pr:.4f}{brier}")


def main():
    df = load_data(DATA_FILE)
    train = df[df["date"] < "2025-01-01"].reset_index(drop=True)
    test = df[df["date"] >= "2025-01-01"].reset_index(drop=True)

    inspect_2025_events(test)
    print("--- TAIL-PRESERVING MOS BENCHMARK (Leak-Free 2024 CV -> 2025) ---")

    # 0. Raw ECMWF
    evaluate("0. Raw ECMWF forecast_mm", train, test, lambda tr, va: va["forecast_rainfall_mm"].values)

    # 1. Calibrated 1-feature LogReg on linear raw forecast (baseline probabilities)
    def raw_calibrated(tr, va):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
        clf.fit(tr[["forecast_rainfall_mm"]], tr["heavy_rain_tomorrow"])
        return clf.predict_proba(va[["forecast_rainfall_mm"]])[:, 1]

    evaluate("1. Linear LogReg (Raw ECMWF only)", train, test, raw_calibrated)

    # 2. Shrunk Location Bias Scaling -> Calibrated Prob
    def shrunk_bias_mos(tr, va):
        ratios, g = fit_location_ratios(tr, prior_mm=300.0)
        tr_corr = (tr["forecast_rainfall_mm"] * tr["location"].map(ratios).fillna(g)).values.reshape(-1, 1)
        va_corr = (va["forecast_rainfall_mm"] * va["location"].map(ratios).fillna(g)).values.reshape(-1, 1)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
        clf.fit(tr_corr, tr["heavy_rain_tomorrow"])
        return clf.predict_proba(va_corr)[:, 1]

    evaluate("2. Shrunk Loc-Bias -> Linear LogReg", train, test, shrunk_bias_mos)

    # 3. Shrunk Location Bias + Regional Other-Cities Rain
    def bias_plus_regional(tr, va):
        ratios, g = fit_location_ratios(tr, prior_mm=300.0)
        X_tr = np.column_stack([
            tr["forecast_rainfall_mm"] * tr["location"].map(ratios).fillna(g),
            tr["other_cities_mean_rain"],
        ])
        X_va = np.column_stack([
            va["forecast_rainfall_mm"] * va["location"].map(ratios).fillna(g),
            va["other_cities_mean_rain"],
        ])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5))
        clf.fit(X_tr, tr["heavy_rain_tomorrow"])
        return clf.predict_proba(X_va)[:, 1]

    evaluate("3. Loc-Bias + Regional Rain", train, test, bias_plus_regional)

    # 4. Shrunk Location Bias + Regional + Prev-Day Persistence
    def bias_reg_persist(tr, va):
        ratios, g = fit_location_ratios(tr, prior_mm=300.0)
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
        clf.fit(X_tr, tr["heavy_rain_tomorrow"])
        return clf.predict_proba(X_va)[:, 1]

    evaluate("4. Loc-Bias + Regional + Prev-Day", train, test, bias_reg_persist)


if __name__ == "__main__":
    main()