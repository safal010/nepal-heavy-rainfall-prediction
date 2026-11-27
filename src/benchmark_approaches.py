from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge, TweedieRegressor
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = Path("data/fixed_run_forecast_training_data.csv")
THRESHOLD_MM = 50.0


def prepare_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "location"]).reset_index(drop=True)

    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["log_forecast_rain"] = np.log1p(df["forecast_rainfall_mm"].clip(lower=0))
    df["log_actual_rain"] = np.log1p(df["actual_rainfall_mm"].clip(lower=0))
    # Group key: ISO year-week prevents same-storm leakage across cities/adjacent days
    df["storm_week"] = df["date"].dt.strftime("%Y-%W")
    return df


def print_location_bias(train: pd.DataFrame):
    """Show ECMWF bias on rainy days (>= 5 mm actual) in 2024."""
    print("--- 2024 PER-LOCATION BIAS (Rainy days: actual >= 5 mm) ---")
    rainy = train[train["actual_rainfall_mm"] >= 5.0]
    summary = (
        rainy.groupby("location")
        .agg(
            rainy_days=("actual_rainfall_mm", "count"),
            heavy_days=("heavy_rain_tomorrow", "sum"),
            mean_ecmwf_mm=("forecast_rainfall_mm", "mean"),
            mean_actual_mm=("actual_rainfall_mm", "mean"),
        )
        .assign(ratio_actual_to_ecmwf=lambda d: d["mean_actual_mm"] / d["mean_ecmwf_mm"])
    )
    print(summary.round(2).to_string())
    print()


def evaluate_oof_and_test(name, train, test, fit_predict_fn):
    cv = StratifiedGroupKFold(n_splits=5)
    y_train_bin = train["heavy_rain_tomorrow"].values
    y_test_bin = test["heavy_rain_tomorrow"].values

    oof_scores = np.zeros(len(train))
    for tr_idx, va_idx in cv.split(train, y_train_bin, groups=train["storm_week"]):
        oof_scores[va_idx] = fit_predict_fn(train.iloc[tr_idx], train.iloc[va_idx])

    test_scores = fit_predict_fn(train, test)

    oof_pr = average_precision_score(y_train_bin, oof_scores)
    test_pr = average_precision_score(y_test_bin, test_scores)

    brier_str = ""
    if 0.0 <= test_scores.min() and test_scores.max() <= 1.0:
        brier_str = f" | 2025 Brier: {brier_score_loss(y_test_bin, test_scores):.4f}"

    print(
        f"{name:34s} | 2024 Leak-Free PR-AUC: {oof_pr:.4f} | "
        f"2025 PR-AUC: {test_pr:.4f}{brier_str}"
    )


def main():
    df = prepare_df(DATA_FILE)
    train = df[df["date"] < "2025-01-01"].reset_index(drop=True)
    test = df[df["date"] >= "2025-01-01"].reset_index(drop=True)

    print_location_bias(train)
    print("--- LEAK-FREE BENCHMARK (Week-Grouped 2024 CV -> Untouched 2025) ---")

    # 0. Raw ECMWF forecast
    evaluate_oof_and_test(
        "0. Raw ECMWF forecast_mm",
        train,
        test,
        lambda tr, va: va["forecast_rainfall_mm"].values,
    )

    # 1. Lean Logistic Regression (only 3 core features, strong regularization)
    def lean_logreg(tr, va):
        cols = ["log_forecast_rain", "humidity_mean_pct", "doy_sin", "doy_cos"]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=2000)),
        ])
        pipe.fit(tr[cols], tr["heavy_rain_tomorrow"])
        return pipe.predict_proba(va[cols])[:, 1]

    evaluate_oof_and_test("1. Lean LogReg (4 features)", train, test, lean_logreg)

    # 2. Log-Rainfall Ridge Regression + Location -> P(rain >= 50 mm)
    def log_ridge_prob(tr, va):
        num_cols = ["log_forecast_rain", "humidity_mean_pct", "temperature_mean_c", "doy_sin", "doy_cos"]
        pre = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("loc", OneHotEncoder(handle_unknown="ignore"), ["location"]),
        ])
        pipe = Pipeline([("pre", pre), ("reg", Ridge(alpha=10.0))])
        pipe.fit(tr, tr["log_actual_rain"])

        # Estimate residual std on rainy training days so dry days don't shrink sigma
        tr_pred = pipe.predict(tr)
        rainy_mask = tr["forecast_rainfall_mm"].values >= 5.0
        sigma = float(np.std(tr["log_actual_rain"].values[rainy_mask] - tr_pred[rainy_mask]))

        va_mu = pipe.predict(va)
        return norm.sf(np.log1p(THRESHOLD_MM), loc=va_mu, scale=max(sigma, 1e-3))

    evaluate_oof_and_test("2. Log-Ridge + Location -> Prob", train, test, log_ridge_prob)

    # 3. Log-Rainfall Ridge trained ONLY on rainy forecasts (>= 1 mm)
    def wet_only_log_ridge(tr, va):
        num_cols = ["log_forecast_rain", "humidity_mean_pct", "doy_sin", "doy_cos"]
        pre = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("loc", OneHotEncoder(handle_unknown="ignore"), ["location"]),
        ])
        wet_tr = tr[tr["forecast_rainfall_mm"] >= 1.0]
        pipe = Pipeline([("pre", pre), ("reg", Ridge(alpha=10.0))])
        pipe.fit(wet_tr, wet_tr["log_actual_rain"])

        tr_pred = pipe.predict(wet_tr)
        sigma = float(np.std(wet_tr["log_actual_rain"].values - tr_pred))

        va_mu = pipe.predict(va)
        probs = norm.sf(np.log1p(THRESHOLD_MM), loc=va_mu, scale=max(sigma, 1e-3))
        # Zero out clearly dry days (< 1 mm forecast)
        return np.where(va["forecast_rainfall_mm"].values < 1.0, 0.0, probs)

    evaluate_oof_and_test("3. Wet-Only Log-Ridge + Loc", train, test, wet_only_log_ridge)

    # 4. Tweedie Regression (built for zero-inflated continuous rainfall)
    def tweedie_reg(tr, va):
        num_cols = ["log_forecast_rain", "humidity_mean_pct", "doy_sin", "doy_cos"]
        pre = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("loc", OneHotEncoder(handle_unknown="ignore"), ["location"]),
        ])
        pipe = Pipeline([
            ("pre", pre),
            ("reg", TweedieRegressor(power=1.5, alpha=1.0, max_iter=1000)),
        ])
        pipe.fit(tr, tr["actual_rainfall_mm"])
        return pipe.predict(va)

    evaluate_oof_and_test("4. TweedieRegressor (power=1.5)", train, test, tweedie_reg)

    # 5. Pure per-location multiplicative bias correction of ECMWF
    def loc_bias_scaling(tr, va):
        rainy = tr[tr["forecast_rainfall_mm"] >= 5.0]
        ratios = (
            rainy.groupby("location")["actual_rainfall_mm"].sum()
            / rainy.groupby("location")["forecast_rainfall_mm"].sum()
        ).to_dict()
        scales = va["location"].map(ratios).fillna(1.0).values
        return va["forecast_rainfall_mm"].values * scales

    evaluate_oof_and_test("5. Location Bias Scaling Only", train, test, loc_bias_scaling)


if __name__ == "__main__":
    main()