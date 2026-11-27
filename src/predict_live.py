"""
predict_live.py — Live 4-day heavy rainfall forecast across all 24 Nepal regions
(7 provinces, Tarai / Hill / Mountain belts) using trained ML models + Open-Meteo
3-model NWP ensemble (ECMWF IFS 0.25°, GFS Seamless, ICON Seamless).

Usage:
    python src/predict_live.py
    python src/predict_live.py --no-log
    python src/predict_live.py --force-notify
    python src/predict_live.py --min-alert-prob 0.35
    python src/predict_live.py --province "Gandaki"
    python src/predict_live.py --alerts-only
"""

from __future__ import annotations

import argparse
import os
import smtplib
import sys
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

# Ensure `src/` is importable whether invoked as `python src/predict_live.py` or `-m src.predict_live`
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nepal_regions import NEPAL_REGIONS  # noqa: E402

CITIES: list[dict[str, Any]] = NEPAL_REGIONS
CITY_META: dict[str, dict[str, Any]] = {c["city"]: c for c in CITIES}

NPT = timezone(timedelta(hours=5, minutes=45))
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "live_predictions.csv"

DAILY_VARS = [
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

HOURLY_VARS = [
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "cloud_cover",
    "cape",
    "precipitation",
]

LOG_COLUMNS = [
    "run_time_utc",
    "run_date_npt",
    "province",
    "belt",
    "calibration",
    "city",
    "date",
    "lead_day",
    "heavy_rain_prob",
    "risk_level",
    "pred_rain_mm",
    "pred_rain_q10",
    "pred_rain_q90",
    "ecmwf_mm",
    "gfs_mm",
    "icon_mm",
    "ens3_mean_mm",
    "ens3_std_mm",
]


def _get_json(url: str, params: dict[str, Any], retries: int = 3) -> Any:
    """HTTP GET with exponential backoff."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=45)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Open-Meteo request failed after {retries} attempts: {last_err}")


def fetch_all_cities_batched(
    cities: list[dict[str, Any]],
    models: str | None = None,
    daily: list[str] | None = None,
    hourly: list[str] | None = None,
    past_days: int = 2,
    forecast_days: int = 5,
    batch_size: int = 25,
) -> list[dict[str, Any]]:
    """Fetch Open-Meteo forecasts for all regions in clean coordinate batches."""
    results: list[dict[str, Any]] = []
    for i in range(0, len(cities), batch_size):
        chunk = cities[i : i + batch_size]
        params: dict[str, Any] = {
            "latitude": ",".join(str(c["lat"]) for c in chunk),
            "longitude": ",".join(str(c["lon"]) for c in chunk),
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "Asia/Kathmandu",
        }
        if models:
            params["models"] = models
        if daily:
            params["daily"] = ",".join(daily)
        if hourly:
            params["hourly"] = ",".join(hourly)

        data = _get_json(FORECAST_URL, params)
        if isinstance(data, dict):
            results.append(data)
        else:
            results.extend(data)
    return results


def fetch_live_weather_frame(cities: list[dict[str, Any]]) -> pd.DataFrame:
    """Fetch 3-model NWP precipitation + atmospheric daily/hourly variables for all regions."""
    # 1. Core daily meteorology (best_match)
    daily_res = fetch_all_cities_batched(
        cities,
        daily=DAILY_VARS,
        past_days=2,
        forecast_days=5,
    )
    daily_frames: list[pd.DataFrame] = []
    for city_meta, item in zip(cities, daily_res):
        d = item["daily"]
        cdf = pd.DataFrame(
            {
                "city": city_meta["city"],
                "province": city_meta["province"],
                "belt": city_meta["belt"],
                "calibration": city_meta["calibration"],
                "lat": city_meta["lat"],
                "lon": city_meta["lon"],
                "elev": city_meta["elev"],
                "date": pd.to_datetime(d["time"]),
                "precip_best": d.get("precipitation_sum"),
                "rain_best": d.get("rain_sum"),
                "precip_hours": d.get("precipitation_hours"),
                "t_max": d.get("temperature_2m_max"),
                "t_min": d.get("temperature_2m_min"),
                "wind_max": d.get("wind_speed_10m_max"),
                "gust_max": d.get("wind_gusts_10m_max"),
                "rad_sum": d.get("shortwave_radiation_sum"),
                "et0": d.get("et0_fao_evapotranspiration"),
            }
        )
        daily_frames.append(cdf)
    df = pd.concat(daily_frames, ignore_index=True)

    # 2. Hourly atmospheric variables aggregated to daily features
    hourly_res = fetch_all_cities_batched(
        cities,
        hourly=HOURLY_VARS,
        past_days=2,
        forecast_days=5,
    )
    hourly_frames: list[pd.DataFrame] = []
    for city_meta, item in zip(cities, hourly_res):
        h = item["hourly"]
        hdf = pd.DataFrame(
            {
                "time": pd.to_datetime(h["time"]),
                "rh": h.get("relative_humidity_2m"),
                "dew": h.get("dew_point_2m"),
                "pressure": h.get("surface_pressure"),
                "cloud": h.get("cloud_cover"),
                "cape": h.get("cape"),
                "h_precip": h.get("precipitation"),
            }
        )
        hdf["date"] = hdf["time"].dt.normalize()
        agg = (
            hdf.groupby("date", as_index=False)
            .agg(
                rh_mean=("rh", "mean"),
                rh_max=("rh", "max"),
                dew_mean=("dew", "mean"),
                dew_max=("dew", "max"),
                pressure_mean=("pressure", "mean"),
                pressure_min=("pressure", "min"),
                cloud_mean=("cloud", "mean"),
                cape_max=("cape", "max"),
                cape_mean=("cape", "mean"),
                hourly_precip_max=("h_precip", "max"),
                hourly_precip_sum=("h_precip", "sum"),
            )
        )
        agg["city"] = city_meta["city"]
        hourly_frames.append(agg)
    df_hourly = pd.concat(hourly_frames, ignore_index=True)
    df = df.merge(df_hourly, on=["city", "date"], how="left")

    # 3. 3-Model NWP precipitation ensemble (ECMWF IFS 0.25°, GFS Seamless, ICON Seamless)
    nwp_res = fetch_all_cities_batched(
        cities,
        models="ecmwf_ifs025,gfs_seamless,icon_seamless",
        daily=["precipitation_sum"],
        past_days=2,
        forecast_days=5,
    )
    nwp_frames: list[pd.DataFrame] = []
    for city_meta, item in zip(cities, nwp_res):
        d = item["daily"]
        nwp_frames.append(
            pd.DataFrame(
                {
                    "city": city_meta["city"],
                    "date": pd.to_datetime(d["time"]),
                    "ecmwf_mm": d.get("precipitation_sum_ecmwf_ifs025"),
                    "gfs_mm": d.get("precipitation_sum_gfs_seamless"),
                    "icon_mm": d.get("precipitation_sum_icon_seamless"),
                }
            )
        )
    df_nwp = pd.concat(nwp_frames, ignore_index=True)
    df = df.merge(df_nwp, on=["city", "date"], how="left")

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the exact model feature set used during training and Streamlit inference."""
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    # Ensemble statistics across ECMWF, GFS, ICON
    nwp_cols = ["ecmwf_mm", "gfs_mm", "icon_mm"]
    for col in nwp_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df["precip_best"])

    df["ens3_mean_mm"] = df[nwp_cols].mean(axis=1)
    df["ens3_std_mm"] = df[nwp_cols].std(axis=1).fillna(0.0)
    df["ens3_max_mm"] = df[nwp_cols].max(axis=1)
    df["ens3_min_mm"] = df[nwp_cols].min(axis=1)
    df["ens3_range_mm"] = df["ens3_max_mm"] - df["ens3_min_mm"]
    df["ens3_n_models_heavy"] = (df[nwp_cols] >= 35.0).sum(axis=1)
    df["ens3_n_models_wet"] = (df[nwp_cols] >= 10.0).sum(axis=1)

    # Model-vs-model differences & ratio
    df["ecmwf_minus_gfs"] = df["ecmwf_mm"] - df["gfs_mm"]
    df["ecmwf_minus_icon"] = df["ecmwf_mm"] - df["icon_mm"]
    df["ecmwf_ratio_ens"] = df["ecmwf_mm"] / (df["ens3_mean_mm"] + 1.0)

    # Cross-Nepal synoptic signal (mean rainfall of all other regions on the same day)
    date_sum = df.groupby("date")["ens3_mean_mm"].transform("sum")
    date_cnt = df.groupby("date")["ens3_mean_mm"].transform("count")
    df["ens3_other_cities_mm"] = np.where(
        date_cnt > 1,
        (date_sum - df["ens3_mean_mm"]) / (date_cnt - 1),
        df["ens3_mean_mm"],
    )

    # Temporal / multi-day synoptic features per region
    g = df.groupby("city", sort=False)
    df["ens3_prev_day_mm"] = g["ens3_mean_mm"].shift(1).fillna(0.0)
    df["ens3_next_day_mm"] = g["ens3_mean_mm"].shift(-1).fillna(0.0)
    df["ens3_3day_sum_mm"] = (
        df["ens3_prev_day_mm"] + df["ens3_mean_mm"] + df["ens3_next_day_mm"]
    )
    df["ens3_max_3day_mm"] = np.maximum.reduce(
        [df["ens3_prev_day_mm"], df["ens3_mean_mm"], df["ens3_next_day_mm"]]
    )
    df["ecmwf_3day_sum_mm"] = (
        g["ecmwf_mm"].shift(1).fillna(0.0)
        + df["ecmwf_mm"]
        + g["ecmwf_mm"].shift(-1).fillna(0.0)
    )
    df["gfs_3day_sum_mm"] = (
        g["gfs_mm"].shift(1).fillna(0.0)
        + df["gfs_mm"]
        + g["gfs_mm"].shift(-1).fillna(0.0)
    )

    # Thermodynamics, pressure trends & wind structure
    df["t_range"] = df["t_max"] - df["t_min"]
    df["t_mean_c"] = 0.5 * (df["t_max"] + df["t_min"])
    df["rh_dew_proxy"] = df["rh_mean"] * df["dew_mean"] / 100.0
    df["pressure_drop_1d"] = df["pressure_mean"] - g["pressure_mean"].shift(1)
    df["pressure_drop_2d"] = df["pressure_mean"] - g["pressure_mean"].shift(2)
    df["pressure_drop_1d"] = df["pressure_drop_1d"].fillna(0.0)
    df["pressure_drop_2d"] = df["pressure_drop_2d"].fillna(0.0)
    df["wind_gust_ratio"] = df["gust_max"] / (df["wind_max"].clip(lower=1.0))

    # Seasonality
    df["doy"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    month = df["date"].dt.month
    df["is_monsoon"] = month.isin([6, 7, 8, 9]).astype(int)
    df["is_peak_monsoon"] = month.isin([7, 8]).astype(int)

    # Ensure cape has no NaNs
    for col in ["cape_max", "cape_mean", "hourly_precip_max", "hourly_precip_sum"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def load_models(models_dir: Path) -> dict[str, Any]:
    """Load trained classifier, regressors, and feature list."""
    meta = joblib.load(models_dir / "model_meta.joblib")
    return {
        "meta": meta,
        "features": meta["features"],
        "clf": joblib.load(models_dir / "heavy_rain_clf.joblib"),
        "reg_mean": joblib.load(models_dir / "rain_mm_reg.joblib"),
        "reg_q10": joblib.load(models_dir / "rain_mm_q10.joblib"),
        "reg_q90": joblib.load(models_dir / "rain_mm_q90.joblib"),
    }


def classify_risk(prob: float) -> str:
    if prob >= 0.60:
        return "HIGH"
    if prob >= 0.35:
        return "MODERATE"
    if prob >= 0.15:
        return "WATCH"
    return "LOW"


def predict_live(
    models_dir: Path = DEFAULT_MODELS_DIR,
    cities: list[dict[str, Any]] = CITIES,
    horizon_days: int = 4,
) -> pd.DataFrame:
    """Run live 4-day inference across all Nepal regions and return tidy predictions."""
    artifacts = load_models(models_dir)
    features: list[str] = artifacts["features"]

    raw_df = fetch_live_weather_frame(cities)
    feat_df = build_features(raw_df)

    now_utc = datetime.now(timezone.utc)
    now_npt = now_utc.astimezone(NPT)
    today_npt = now_npt.date()
    end_date = today_npt + timedelta(days=horizon_days - 1)

    mask = (feat_df["date"].dt.date >= today_npt) & (feat_df["date"].dt.date <= end_date)
    pred_df = feat_df.loc[mask].copy().reset_index(drop=True)

    # Align feature columns cleanly (defaulting any missing optional column to 0.0)
    X = pd.DataFrame(index=pred_df.index)
    for col in features:
        if col in pred_df.columns:
            X[col] = pd.to_numeric(pred_df[col], errors="coerce").fillna(0.0)
        else:
            X[col] = 0.0

    # Predictions
    heavy_prob = artifacts["clf"].predict_proba(X)[:, 1]
    pred_mm = np.expm1(np.clip(artifacts["reg_mean"].predict(X), 0.0, None))
    pred_q10 = np.expm1(np.clip(artifacts["reg_q10"].predict(X), 0.0, None))
    pred_q90 = np.expm1(np.clip(artifacts["reg_q90"].predict(X), 0.0, None))

    # Dry-day sanity floor: when all 3 NWP models agree on <= 2 mm, prevent false alarms
    dry_mask = (
        (pred_df["ecmwf_mm"].fillna(0.0) <= 2.0)
        & (pred_df["gfs_mm"].fillna(0.0) <= 2.0)
        & (pred_df["icon_mm"].fillna(0.0) <= 2.0)
    ).to_numpy()

    heavy_prob = np.where(dry_mask, np.minimum(heavy_prob, 0.05), heavy_prob)
    pred_mm = np.where(dry_mask, np.minimum(pred_mm, 3.0), pred_mm)
    pred_q10 = np.where(dry_mask, np.minimum(pred_q10, 0.5), pred_q10)
    pred_q90 = np.where(dry_mask, np.minimum(pred_q90, 5.0), pred_q90)

    # Enforce monotonic quantiles: 0 <= q10 <= pred_mm <= q90
    pred_mm = np.clip(pred_mm, 0.0, None)
    pred_q10 = np.clip(pred_q10, 0.0, pred_mm)
    pred_q90 = np.maximum(pred_q90, pred_mm)

    out = pd.DataFrame(
        {
            "run_time_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_date_npt": today_npt.isoformat(),
            "province": pred_df["province"],
            "belt": pred_df["belt"],
            "calibration": pred_df["calibration"],
            "city": pred_df["city"],
            "date": pred_df["date"].dt.strftime("%Y-%m-%d"),
            "lead_day": pred_df["date"].dt.date.apply(lambda d: (d - today_npt).days),
            "heavy_rain_prob": np.round(heavy_prob, 4),
            "risk_level": [classify_risk(p) for p in heavy_prob],
            "pred_rain_mm": np.round(pred_mm, 1),
            "pred_rain_q10": np.round(pred_q10, 1),
            "pred_rain_q90": np.round(pred_q90, 1),
            "ecmwf_mm": np.round(pred_df["ecmwf_mm"].fillna(0.0), 1),
            "gfs_mm": np.round(pred_df["gfs_mm"].fillna(0.0), 1),
            "icon_mm": np.round(pred_df["icon_mm"].fillna(0.0), 1),
            "ens3_mean_mm": np.round(pred_df["ens3_mean_mm"].fillna(0.0), 1),
            "ens3_std_mm": np.round(pred_df["ens3_std_mm"].fillna(0.0), 1),
        }
    )

    out = out.sort_values(
        ["date", "heavy_rain_prob", "pred_rain_mm"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return out


def append_to_log(df: pd.DataFrame, log_path: Path = DEFAULT_LOG_PATH) -> Path:
    """Append live predictions to CSV log, deduplicating on (run_date_npt, city, date)
    and backfilling province/belt/calibration for any legacy 6-city rows."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists() and log_path.stat().st_size > 0:
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.copy()

    # Backfill metadata columns for older log rows created before 24-region expansion
    for meta_col in ("province", "belt", "calibration"):
        if meta_col not in combined.columns:
            combined[meta_col] = np.nan
        mapped = combined["city"].map(lambda c: CITY_META.get(c, {}).get(meta_col, "Unknown"))
        combined[meta_col] = combined[meta_col].fillna(mapped)

    combined = combined.drop_duplicates(
        subset=["run_date_npt", "city", "date"],
        keep="last",
    )

    ordered_cols = [c for c in LOG_COLUMNS if c in combined.columns] + [
        c for c in combined.columns if c not in LOG_COLUMNS
    ]
    combined = combined[ordered_cols].sort_values(
        ["run_date_npt", "date", "province", "city"]
    )
    combined.to_csv(log_path, index=False)
    return log_path


def build_alert_message(
    df: pd.DataFrame,
    min_alert_prob: float = 0.35,
    force_notify: bool = False,
) -> tuple[str | None, pd.DataFrame]:
    """Build a concise alert digest grouped by date for regions exceeding min_alert_prob."""
    alerts = df[df["heavy_rain_prob"] >= min_alert_prob].copy()
    if alerts.empty and not force_notify:
        return None, alerts

    run_date = df["run_date_npt"].iloc[0] if not df.empty else datetime.now(NPT).date().isoformat()
    n_high = int((df["risk_level"] == "HIGH").sum())
    n_mod = int((df["risk_level"] == "MODERATE").sum())
    n_watch = int((df["risk_level"] == "WATCH").sum())

    lines = [
        f"🌧️ Nepal Heavy Rain Forecast Alert ({run_date} NPT)",
        f"Coverage: {df['city'].nunique()} regions / {df['province'].nunique()} provinces | "
        f"HIGH: {n_high} | MODERATE: {n_mod} | WATCH: {n_watch}",
        "",
    ]

    if alerts.empty:
        top = df.sort_values(["heavy_rain_prob", "pred_rain_mm"], ascending=False).head(5)
        lines.append(f"No region-days crossed the {min_alert_prob:.0%} alert threshold.")
        lines.append("Top 5 wettest region-days:")
        for _, r in top.iterrows():
            lines.append(
                f"• {r['city']} ({r['province']} | {r['belt']}) — {r['date']} (D+{r['lead_day']}): "
                f"{r['heavy_rain_prob']:.0%} [{r['risk_level']}] | "
                f"ML {r['pred_rain_mm']:.1f} mm (NWP {r['ens3_mean_mm']:.1f} mm)"
            )
    else:
        alerts = alerts.sort_values(
            ["date", "heavy_rain_prob", "pred_rain_mm"],
            ascending=[True, False, False],
        )
        for date_str, grp in alerts.groupby("date", sort=True):
            lead = int(grp["lead_day"].iloc[0])
            lines.append(f"📅 {date_str} (D+{lead}) — {len(grp)} alert(s):")
            for _, r in grp.iterrows():
                icon = "🔴" if r["risk_level"] == "HIGH" else "🟠"
                lines.append(
                    f"  {icon} {r['city']} ({r['province']} | {r['belt']}): "
                    f"{r['heavy_rain_prob']:.0%} [{r['risk_level']}] | "
                    f"ML {r['pred_rain_mm']:.0f} mm [{r['pred_rain_q10']:.0f}–{r['pred_rain_q90']:.0f}] "
                    f"(ECMWF {r['ecmwf_mm']:.0f} / GFS {r['gfs_mm']:.0f} / ICON {r['icon_mm']:.0f})"
                )
            lines.append("")

    return "\n".join(lines).strip(), alerts


def send_telegram(message: str, max_chars: int = 3900) -> bool:
    """Send alert via Telegram Bot API, auto-chunking messages longer than max_chars."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False

    # Split cleanly on newlines so nationwide monsoon digests never exceed 4,096 chars
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0
    for line in message.splitlines():
        add_len = len(line) + 1
        if current_lines and current_len + add_len > max_chars:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = len(line)
        else:
            current_lines.append(line)
            current_len += add_len
    if current_lines:
        chunks.append("\n".join(current_lines))

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for idx, chunk in enumerate(chunks, start=1):
        text = f"[{idx}/{len(chunks)}]\n{chunk}" if len(chunks) > 1 else chunk
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=20,
        )
        resp.raise_for_status()
    return True


def send_email(subject: str, body: str) -> bool:
    """Send alert via SMTP if ALERT_SMTP_* environment variables are configured."""
    host = os.getenv("ALERT_SMTP_HOST")
    port = int(os.getenv("ALERT_SMTP_PORT", "587"))
    user = os.getenv("ALERT_SMTP_USER")
    password = os.getenv("ALERT_SMTP_PASS")
    to_addr = os.getenv("ALERT_EMAIL_TO")

    if not all([host, user, password, to_addr]):
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=20) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
    return True


def print_forecast_table(
    df: pd.DataFrame,
    province: str | None = None,
    alerts_only: bool = False,
    min_alert_prob: float = 0.35,
) -> None:
    """Pretty-print live forecast summary to stdout."""
    view = df.copy()
    if province:
        view = view[view["province"].str.lower() == province.strip().lower()]
    if alerts_only:
        view = view[view["heavy_rain_prob"] >= min_alert_prob]

    run_time = df["run_time_utc"].iloc[0]
    run_date = df["run_date_npt"].iloc[0]
    risk_counts = df["risk_level"].value_counts().to_dict()

    print("=" * 108)
    print(f"NEPAL LIVE 4-DAY HEAVY RAINFALL FORECAST  |  Run: {run_time} ({run_date} NPT)")
    print(
        f"Regions: {df['city'].nunique()} across {df['province'].nunique()} provinces  |  "
        f"HIGH: {risk_counts.get('HIGH', 0)}  |  "
        f"MODERATE: {risk_counts.get('MODERATE', 0)}  |  "
        f"WATCH: {risk_counts.get('WATCH', 0)}  |  "
        f"LOW: {risk_counts.get('LOW', 0)}"
    )
    print("=" * 108)

    if view.empty:
        print("No rows match the selected display filter.")
        return

    display_cols = [
        "date",
        "lead_day",
        "province",
        "belt",
        "city",
        "heavy_rain_prob",
        "risk_level",
        "pred_rain_mm",
        "ecmwf_mm",
        "gfs_mm",
        "icon_mm",
        "ens3_mean_mm",
    ]
    fmt = view[display_cols].copy()
    fmt["heavy_rain_prob"] = fmt["heavy_rain_prob"].map(lambda x: f"{x * 100:5.1f}%")
    fmt = fmt.rename(
        columns={
            "lead_day": "D+",
            "heavy_rain_prob": "Prob",
            "risk_level": "Risk",
            "pred_rain_mm": "ML_mm",
            "ecmwf_mm": "ECMWF",
            "gfs_mm": "GFS",
            "icon_mm": "ICON",
            "ens3_mean_mm": "NWP_Avg",
        }
    )

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 140,
    ):
        print(fmt.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live 4-day heavy rainfall forecast across all 24 Nepal regions."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Path to directory containing trained *.joblib artifacts.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="CSV path for logging daily predictions.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip writing results to logs/live_predictions.csv.",
    )
    parser.add_argument(
        "--force-notify",
        action="store_true",
        help="Send Telegram/email summary even if no region crosses --min-alert-prob.",
    )
    parser.add_argument(
        "--min-alert-prob",
        type=float,
        default=0.35,
        help="Minimum heavy-rain probability to trigger alerts (default: 0.35).",
    )
    parser.add_argument(
        "--province",
        type=str,
        default=None,
        help="Filter terminal table to one province (full 24-region run is still logged).",
    )
    parser.add_argument(
        "--alerts-only",
        action="store_true",
        help="Only print rows at or above --min-alert-prob in the terminal table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = predict_live(models_dir=args.models_dir, cities=CITIES)

    print_forecast_table(
        df,
        province=args.province,
        alerts_only=args.alerts_only,
        min_alert_prob=args.min_alert_prob,
    )

    if not args.no_log:
        saved_path = append_to_log(df, log_path=args.log_path)
        print(f"\nSaved {len(df)} predictions to {saved_path}")

    msg, alerts = build_alert_message(
        df,
        min_alert_prob=args.min_alert_prob,
        force_notify=args.force_notify,
    )
    if msg:
        tg_sent = send_telegram(msg)
        email_sent = send_email(
            subject=f"Nepal Rain Alert ({df['run_date_npt'].iloc[0]}): {len(alerts)} alert(s)",
            body=msg,
        )
        print(
            f"Alerts triggered: {len(alerts)} | "
            f"Telegram sent: {tg_sent} | Email sent: {email_sent}"
        )
    else:
        print(f"No alerts triggered (threshold >= {args.min_alert_prob:.0%}).")


if __name__ == "__main__":
    main()