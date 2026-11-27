import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path("data/fixed_run_forecast_training_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create seasonality and multi-day persistence features per location."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    # Monsoon seasonality
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Non-linear rainfall scale (helps linear models)
    df["log_forecast_rain"] = np.log1p(df["forecast_rainfall_mm"].clip(lower=0))

    # Multi-day persistence within each location
    grp = df.groupby("location")["forecast_rainfall_mm"]
    df["rain_lag1"] = grp.shift(1).fillna(0.0)
    df["rain_lag2"] = grp.shift(2).fillna(0.0)
    df["rain_3d_sum"] = (
        df["forecast_rainfall_mm"] + df["rain_lag1"] + df["rain_lag2"]
    )

    return df


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Pick the probability threshold that maximizes F1 on OOF training predictions."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1s = (2 * precisions[:-1] * recalls[:-1]) / np.clip(
        precisions[:-1] + recalls[:-1], 1e-12, None
    )
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def main():
    df = pd.read_csv(DATA_FILE)
    df = add_features(df)

    features = [
        "latitude",
        "longitude",
        "temperature_mean_c",
        "humidity_mean_pct",
        "forecast_rainfall_mm",
        "wind_speed_mean_kmh",
        "doy_sin",
        "doy_cos",
        "log_forecast_rain",
        "rain_lag1",
        "rain_lag2",
        "rain_3d_sum",
    ]
    target = "heavy_rain_tomorrow"

    train = df[df["date"] < "2025-01-01"].copy()
    test = df[df["date"] >= "2025-01-01"].copy()

    X_train, y_train = train[features], train[target].values
    X_test, y_test = test[features], test[target].values

    print(f"Train: {len(train)} rows ({y_train.sum()} heavy-rain events)")
    print(f"Test:  {len(test)} rows ({y_test.sum()} heavy-rain events)\n")

    # 1. Raw ECMWF forecast baseline (no ML)
    raw_pr_auc = average_precision_score(y_test, test["forecast_rainfall_mm"])
    print(f"[Baseline] Raw ECMWF forecast_rainfall_mm | 2025 PR-AUC: {raw_pr_auc:.4f}\n")

    # 2. Candidate models (calibrated, no class_weight distortion)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2000, random_state=42),
        ),
        "HistGradientBoosting": CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_depth=3,
                min_samples_leaf=20,
                l2_regularization=1.0,
                learning_rate=0.05,
                max_iter=150,
                random_state=42,
            ),
            method="sigmoid",
            cv=3,
        ),
        "RandomForest": CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
            method="sigmoid",
            cv=3,
        ),
    }

    best_name = None
    best_oof_pr_auc = -1.0
    best_threshold = 0.5
    best_estimator = None

    print("--- 2024 OUT-OF-FOLD BENCHMARK ---")
    for name, est in candidates.items():
        oof_probs = cross_val_predict(
            est, X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        oof_pr_auc = average_precision_score(y_train, oof_probs)
        oof_brier = brier_score_loss(y_train, oof_probs)
        thresh, oof_f1 = find_best_threshold(y_train, oof_probs)

        print(
            f"{name:22s} | OOF PR-AUC: {oof_pr_auc:.4f} | "
            f"Brier: {oof_brier:.4f} | Best Thresh: {thresh:.3f} (OOF F1: {oof_f1:.3f})"
        )

        if oof_pr_auc > best_oof_pr_auc:
            best_oof_pr_auc = oof_pr_auc
            best_name = name
            best_threshold = thresh
            best_estimator = est

    # 3. Fit winning model on all 2024 data and evaluate ONCE on untouched 2025
    print(f"\nSelected model: {best_name} (threshold={best_threshold:.3f})")
    best_estimator.fit(X_train, y_train)

    test_probs = best_estimator.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    test_pr_auc = average_precision_score(y_test, test_probs)
    test_brier = brier_score_loss(y_test, test_probs)

    print("\n--- UNTOUCHED 2025 TEST RESULTS ---")
    print(f"2025 PR-AUC:      {test_pr_auc:.4f} (vs Raw ECMWF: {raw_pr_auc:.4f})")
    print(f"2025 Brier Score: {test_brier:.4f} (lower is better)")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_preds, labels=[0, 1]))
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_preds,
            labels=[0, 1],
            target_names=["Not heavy rain", "Heavy rain"],
            zero_division=0,
        )
    )

    # 4. Save model, features, and threshold metadata
    joblib.dump(best_estimator, MODEL_DIR / "fixed_run_heavy_rain_model.pkl")
    joblib.dump(features, MODEL_DIR / "fixed_run_model_features.pkl")
    with open(MODEL_DIR / "fixed_run_model_meta.json", "w") as f:
        json.dump(
            {
                "model_name": best_name,
                "threshold": round(best_threshold, 4),
                "oof_pr_auc": round(best_oof_pr_auc, 4),
                "test_pr_auc": round(test_pr_auc, 4),
                "raw_ecmwf_test_pr_auc": round(raw_pr_auc, 4),
            },
            f,
            indent=2,
        )
    print("Saved model, features, and metadata to models/.")


if __name__ == "__main__":
    main()