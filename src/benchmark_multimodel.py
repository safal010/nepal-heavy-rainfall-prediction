from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path("data/multimodel_nepal_rain.csv")
MODELS_3 = ["ecmwf", "gfs", "icon"]
MODELS_4 = ["ecmwf", "gfs", "icon", "gem"]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["location", "date"]).reset_index(drop=True)
    df["storm_week"] = df["date"].dt.strftime("%Y-%W")

    # Equal-weight 3-model raw ensemble for regional & persistence features
    df["ens3_raw_mm"] = df[[f"{m}_rain_mm" for m in MODELS_3]].mean(axis=1)
    df["ens3_max_3h_mm"] = df[[f"{m}_max_3h_mm" for m in MODELS_3]].mean(axis=1)
    df["ens3_dew_mean_c"] = df[[f"{m}_dew_mean_c" for m in MODELS_3]].mean(axis=1)

    for prefix, col in [("ecmwf", "ecmwf_rain_mm"), ("ens3", "ens3_raw_mm")]:
        day_sum = df.groupby("date")[col].transform("sum")
        day_cnt = df.groupby("date")[col].transform("count")
        df[f"{prefix}_other_cities_mm"] = (day_sum - df[col]) / np.maximum(day_cnt - 1, 1)
        df[f"{prefix}_prev_day_mm"] = df.groupby("location")[col].shift(1).fillna(0.0)

    return df


def print_multimodel_bias(train: pd.DataFrame):
    print("--- 2024 PER-LOCATION BIAS RATIOS (mean ERA5 / mean Model, where ERA5 >= 5 mm) ---")
    rainy = train[train["actual_era5_mm"] >= 5.0]
    rows = []
    for loc, grp in rainy.groupby("location"):
        rec = {"location": loc, "rainy_days": len(grp), "heavy_50": int((grp["actual_era5_mm"] >= 50).sum())}
        for m in MODELS_4:
            rec[f"{m}_ratio"] = grp["actual_era5_mm"].sum() / max(grp[f"{m}_rain_mm"].sum(), 1e-6)
        rows.append(rec)
    print(pd.DataFrame(rows).set_index("location").round(2).to_string())


def fit_model_ratios(tr: pd.DataFrame, models: list[str], prior_mm: float = 300.0) -> dict:
    """Empirical-Bayes shrunk per-location bias ratio for each NWP model (fit on train fold only)."""
    out = {}
    for m in models:
        col = f"{m}_rain_mm"
        rainy = tr[tr[col] >= 5.0]
        g = rainy["actual_era5_mm"].sum() / max(rainy[col].sum(), 1e-6)
        ratios = {
            loc: (grp["actual_era5_mm"].sum() + prior_mm * g) / (grp[col].sum() + prior_mm)
            for loc, grp in rainy.groupby("location")
        }
        out[m] = (ratios, g)
    return out


def apply_bias(df: pd.DataFrame, ratio_map: dict, models: list[str]) -> np.ndarray:
    cols = []
    for m in models:
        ratios, g = ratio_map[m]
        cols.append((df[f"{m}_rain_mm"] * df["location"].map(ratios).fillna(g)).values)
    return np.column_stack(cols)


def run_benchmark(df: pd.DataFrame, target_col: str, label: str):
    train = df[df["date"] < "2025-01-01"].reset_index(drop=True)
    test = df[df["date"] >= "2025-01-01"].reset_index(drop=True)
    y_tr, y_te = train[target_col].values, test[target_col].values
    cv = StratifiedGroupKFold(n_splits=5)

    print(
        f"\n=== {label} (2024: {int(y_tr.sum())} events, 2025: {int(y_te.sum())} events) ==="
    )

    def eval_fn(name, fn):
        oof = np.zeros(len(train))
        for tr_i, va_i in cv.split(train, y_tr, groups=train["storm_week"]):
            oof[va_i] = fn(train.iloc[tr_i], train.iloc[va_i])
        preds = fn(train, test)
        oof_pr = average_precision_score(y_tr, oof)
        te_pr = average_precision_score(y_te, preds)
        brier = (
            f" | 2025 Brier: {brier_score_loss(y_te, preds):.4f}"
            if (preds.min() >= 0.0 and preds.max() <= 1.0)
            else ""
        )
        print(f"  {name:38s} | 2024 CV: {oof_pr:.4f} | 2025 PR-AUC: {te_pr:.4f}{brier}")

    # 0. Standalone raw models
    for m in MODELS_4:
        eval_fn(f"0. Raw {m.upper()} rain_mm", lambda tr, va, m=m: va[f"{m}_rain_mm"].values)

    # 1. Unsupervised equal-weight 3-model average
    eval_fn("1. Raw 3-Model Mean (E+G+I)", lambda tr, va: va["ens3_raw_mm"].values)

    # 2. Previous best single-model baseline (ECMWF MOS)
    def single_ecmwf_mos(tr, va):
        rmap = fit_model_ratios(tr, ["ecmwf"])
        X_tr = np.column_stack([
            apply_bias(tr, rmap, ["ecmwf"]),
            tr["ecmwf_other_cities_mm"],
            tr["ecmwf_prev_day_mm"],
        ])
        X_va = np.column_stack([
            apply_bias(va, rmap, ["ecmwf"]),
            va["ecmwf_other_cities_mm"],
            va["ecmwf_prev_day_mm"],
        ])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5))
        clf.fit(X_tr, tr[target_col])
        return clf.predict_proba(X_va)[:, 1]

    eval_fn("2. Single-Model MOS (ECMWF only)", single_ecmwf_mos)

    # 3. Bias-corrected equal 3-model blend -> calibrated LogReg
    def bias_equal_3m(tr, va):
        rmap = fit_model_ratios(tr, MODELS_3)
        tr_blend = apply_bias(tr, rmap, MODELS_3).mean(axis=1, keepdims=True)
        va_blend = apply_bias(va, rmap, MODELS_3).mean(axis=1, keepdims=True)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
        clf.fit(tr_blend, tr[target_col])
        return clf.predict_proba(va_blend)[:, 1]

    eval_fn("3. Bias-Corr Equal 3-Model Blend", bias_equal_3m)

    # 4 & 5. Joint LogReg MOS on bias-corrected models + regional + prev-day
    def make_logreg_mos(models: list[str], extra_cols: list[str] | None = None):
        def fn(tr, va):
            rmap = fit_model_ratios(tr, models)
            base_tr = [apply_bias(tr, rmap, models), tr[["ens3_other_cities_mm", "ens3_prev_day_mm"]].values]
            base_va = [apply_bias(va, rmap, models), va[["ens3_other_cities_mm", "ens3_prev_day_mm"]].values]
            if extra_cols:
                base_tr.append(tr[extra_cols].values)
                base_va.append(va[extra_cols].values)
            X_tr, X_va = np.hstack(base_tr), np.hstack(base_va)
            clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.25))
            clf.fit(X_tr, tr[target_col])
            return clf.predict_proba(X_va)[:, 1]
        return fn

    eval_fn("4. 3-Model MOS (LogReg + Reg + Prev)", make_logreg_mos(MODELS_3))
    eval_fn("5. 4-Model MOS (LogReg + Reg + Prev)", make_logreg_mos(MODELS_4))

    # 6 & 7. Continuous Non-Negative Ridge on raw ERA5 mm -> Calibrated Prob
    def make_ridge_mos(models: list[str]):
        def fn(tr, va):
            rmap = fit_model_ratios(tr, models)
            X_tr = np.hstack([
                apply_bias(tr, rmap, models),
                tr[["ens3_other_cities_mm", "ens3_prev_day_mm"]].values,
            ])
            X_va = np.hstack([
                apply_bias(va, rmap, models),
                va[["ens3_other_cities_mm", "ens3_prev_day_mm"]].values,
            ])
            # Fit continuous rainfall weights on ALL 1,752 days (linear mm, no log1p)
            reg = Ridge(alpha=500.0, positive=True)
            reg.fit(X_tr, tr["actual_era5_mm"].values)
            tr_pred_mm = reg.predict(X_tr).reshape(-1, 1)
            va_pred_mm = reg.predict(X_va).reshape(-1, 1)

            cal = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
            cal.fit(tr_pred_mm, tr[target_col])
            return cal.predict_proba(va_pred_mm)[:, 1]
        return fn

    eval_fn("6. Continuous Ridge MOS (3-Model)", make_ridge_mos(MODELS_3))
    eval_fn("7. Continuous Ridge MOS (4-Model)", make_ridge_mos(MODELS_4))

    # 8. Add peak 3-hour intensity & convective features (CAPE + dew point)
    eval_fn(
        "8. 3-Model MOS + Peak3h + CAPE + Dew",
        make_logreg_mos(MODELS_3, ["ens3_max_3h_mm", "gfs_cape_max", "ens3_dew_mean_c"]),
    )

    # 9. Regularized HGB on 3-Model bias-corrected + peak + convective features
    def hgb_multimodel(tr, va):
        rmap = fit_model_ratios(tr, MODELS_3)
        X_tr = np.hstack([
            apply_bias(tr, rmap, MODELS_3),
            tr[["ens3_other_cities_mm", "ens3_prev_day_mm", "ens3_max_3h_mm", "gfs_cape_max", "ens3_dew_mean_c"]].values,
        ])
        X_va = np.hstack([
            apply_bias(va, rmap, MODELS_3),
            va[["ens3_other_cities_mm", "ens3_prev_day_mm", "ens3_max_3h_mm", "gfs_cape_max", "ens3_dew_mean_c"]].values,
        ])
        clf = HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=20, l2_regularization=10.0,
            learning_rate=0.04, max_iter=150, random_state=42,
        )
        clf.fit(X_tr, tr[target_col])
        return clf.predict_proba(X_va)[:, 1]

    eval_fn("9. Regularized HGB (3-Model + Peak)", hgb_multimodel)

    # Print learned continuous Ridge weights on full 2024 training data
    if target_col == "heavy_rain_50mm":
        rmap = fit_model_ratios(train, MODELS_4)
        X_tr = np.hstack([
            apply_bias(train, rmap, MODELS_4),
            train[["ens3_other_cities_mm", "ens3_prev_day_mm"]].values,
        ])
        reg = Ridge(alpha=500.0, positive=True).fit(X_tr, train["actual_era5_mm"].values)
        names = [f"{m}_bias_corr" for m in MODELS_4] + ["ens3_other_cities", "ens3_prev_day"]
        print("\n--- LEARNED 2024 CONTINUOUS RIDGE WEIGHTS (predicting actual ERA5 mm) ---")
        for n, w in zip(names, reg.coef_):
            print(f"  {n:20s}: {w:.4f}")
        print(f"  {'intercept':20s}: {reg.intercept_:.4f}")


def main():
    df = load_data(DATA_FILE)
    train = df[df["date"] < "2025-01-01"]
    print_multimodel_bias(train)
    run_benchmark(df, "heavy_rain_50mm", "TARGET: ERA5 >= 50 mm (Heavy Rain)")
    run_benchmark(df, "heavy_rain_40mm", "TARGET: ERA5 >= 40 mm (Sensitivity Check)")


if __name__ == "__main__":
    main()