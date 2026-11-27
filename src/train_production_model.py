from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path("data/multimodel_nepal_rain.csv")
MODEL_OUT = Path("models/nepal_heavy_rain_mos.joblib")
MODELS_3 = ["ecmwf", "gfs", "icon"]
PRIOR_MM = 300.0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    df["ens3_raw_mm"] = df[[f"{m}_rain_mm" for m in MODELS_3]].mean(axis=1)
    df["ens3_max_3h_mm"] = df[[f"{m}_max_3h_mm" for m in MODELS_3]].mean(axis=1)
    df["ens3_dew_mean_c"] = df[[f"{m}_dew_mean_c" for m in MODELS_3]].mean(axis=1)

    day_sum = df.groupby("date")["ens3_raw_mm"].transform("sum")
    day_cnt = df.groupby("date")["ens3_raw_mm"].transform("count")
    df["ens3_other_cities_mm"] = (day_sum - df["ens3_raw_mm"]) / np.maximum(day_cnt - 1, 1)
    df["ens3_prev_day_mm"] = df.groupby("location")["ens3_raw_mm"].shift(1).fillna(0.0)
    return df


def fit_ratio_map(tr: pd.DataFrame, prior_mm: float = PRIOR_MM) -> dict:
    ratio_map = {}
    for m in MODELS_3:
        col = f"{m}_rain_mm"
        rainy = tr[tr[col] >= 5.0]
        global_ratio = float(rainy["actual_era5_mm"].sum() / max(rainy[col].sum(), 1e-6))
        per_city = {
            loc: float((grp["actual_era5_mm"].sum() + prior_mm * global_ratio) / (grp[col].sum() + prior_mm))
            for loc, grp in rainy.groupby("location")
        }
        ratio_map[m] = {"per_city": per_city, "global": global_ratio}
    return ratio_map


def build_feature_matrices(df: pd.DataFrame, ratio_map: dict) -> tuple[np.ndarray, np.ndarray]:
    bias_cols = []
    for m in MODELS_3:
        per_city = ratio_map[m]["per_city"]
        g = ratio_map[m]["global"]
        scaled = df[f"{m}_rain_mm"] * df["location"].map(per_city).fillna(g)
        bias_cols.append(scaled.values)

    X_reg = np.column_stack(bias_cols + [
        df["ens3_other_cities_mm"].values,
        df["ens3_prev_day_mm"].values,
    ])
    X_clf = np.column_stack([
        X_reg,
        df["ens3_max_3h_mm"].values,
        df["gfs_cape_max"].values,
        df["ens3_dew_mean_c"].values,
    ])
    return X_clf, X_reg


def fit_models(tr: pd.DataFrame):
    ratio_map = fit_ratio_map(tr)
    X_clf, X_reg = build_feature_matrices(tr, ratio_map)

    clf_50 = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.25, max_iter=1000),
    )
    clf_50.fit(X_clf, tr["heavy_rain_50mm"].values)

    reg_mm = make_pipeline(
        SimpleImputer(strategy="median"),
        Ridge(alpha=500.0, positive=True),
    )
    reg_mm.fit(X_reg, tr["actual_era5_mm"].values)
    return ratio_map, clf_50, reg_mm


def main():
    df = engineer_features(pd.read_csv(DATA_FILE))
    train = df[df["date"] < "2025-01-01"].reset_index(drop=True)
    test = df[df["date"] >= "2025-01-01"].reset_index(drop=True)

    # 1. Strict 2024 -> 2025 Holdout Verification
    rmap_val, clf_val, reg_val = fit_models(train)
    X_te_clf, X_te_reg = build_feature_matrices(test, rmap_val)
    probs = clf_val.predict_proba(X_te_clf)[:, 1]
    pred_mm = np.clip(reg_val.predict(X_te_reg), 0.0, None)
    y_te = test["heavy_rain_50mm"].values

    pr_auc = average_precision_score(y_te, probs)
    roc_auc = roc_auc_score(y_te, probs)
    brier = brier_score_loss(y_te, probs)
    mae_mm = float(np.mean(np.abs(test["actual_era5_mm"].values - pred_mm)))

    print("=== UNTOUCHED 2025 HOLDOUT REPORT (Trained on 2024 Only) ===")
    print(f"  PR-AUC (>=50mm) : {pr_auc:.4f}")
    print(f"  ROC-AUC (>=50mm): {roc_auc:.4f}")
    print(f"  Brier Score     : {brier:.4f}")
    print(f"  Mean Prob vs Obs: {probs.mean():.4f} vs {y_te.mean():.4f}")
    print(f"  Expected mm MAE : {mae_mm:.2f} mm\n")

    print("--- 2025 OPERATIONAL ALERT THRESHOLDS (>= 50 mm) ---")
    print(f"{'Threshold':>10s} | {'Alerts':>6s} | {'Hits':>4s}/{int(y_te.sum()):<2d} | {'Precision':>9s} | {'Recall':>7s} | {'False Alarms':>12s}")
    for thr in [0.10, 0.15, 0.25, 0.35, 0.50]:
        pred_bin = (probs >= thr).astype(int)
        hits = int(((pred_bin == 1) & (y_te == 1)).sum())
        fa = int(((pred_bin == 1) & (y_te == 0)).sum())
        prec = precision_score(y_te, pred_bin, zero_division=0)
        rec = recall_score(y_te, pred_bin, zero_division=0)
        print(f"  >= {thr:4.0%}  | {int(pred_bin.sum()):6d} | {hits:4d}/{int(y_te.sum()):<2d} | {prec:8.1%} | {rec:6.1%} | {fa:12d}")

    print("\n--- 2025 PER-CITY PERFORMANCE (Watch >= 15%, Warning >= 35%) ---")
    test_eval = test[["location", "actual_era5_mm", "heavy_rain_50mm"]].copy()
    test_eval["prob_50mm"] = probs
    test_eval["pred_mm"] = pred_mm
    rows = []
    for loc, g in test_eval.groupby("location"):
        ev = int(g["heavy_rain_50mm"].sum())
        w15 = (g["prob_50mm"] >= 0.15).astype(int)
        w35 = (g["prob_50mm"] >= 0.35).astype(int)
        rows.append({
            "location": loc,
            "events_50mm": ev,
            "watch15_hits": f"{int(((w15 == 1) & (g['heavy_rain_50mm'] == 1)).sum())}/{ev}",
            "watch15_alerts": int(w15.sum()),
            "warn35_hits": f"{int(((w35 == 1) & (g['heavy_rain_50mm'] == 1)).sum())}/{ev}",
            "warn35_alerts": int(w35.sum()),
            "pr_auc": round(average_precision_score(g["heavy_rain_50mm"], g["prob_50mm"]), 3) if ev > 0 else np.nan,
        })
    print(pd.DataFrame(rows).set_index("location").to_string())

    # 2. Refit Final Production Artifact on ALL Data (2024 + 2025)
    ratio_map_full, clf_full, reg_full = fit_models(df)
    cities = (
        df[["location", "latitude", "longitude"]]
        .drop_duplicates()
        .sort_values("location")
        .to_dict(orient="records")
    )

    artifact = {
        "models": MODELS_3,
        "ratio_map": ratio_map_full,
        "clf_50mm": clf_full,
        "reg_mm": reg_full,
        "cities": cities,
        "thresholds": {"watch": 0.15, "warning": 0.35},
        "train_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "train_rows": int(len(df)),
        "train_events_50mm": int(df["heavy_rain_50mm"].sum()),
        "holdout_2025_metrics": {
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "brier": float(brier),
            "mae_mm": mae_mm,
        },
    }

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUT)
    print(f"\nRefitted on {len(df)} rows ({int(df['heavy_rain_50mm'].sum())} events) -> saved {MODEL_OUT}")


if __name__ == "__main__":
    main()