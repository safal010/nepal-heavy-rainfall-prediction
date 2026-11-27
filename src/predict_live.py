import argparse
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import os
from pathlib import Path
import smtplib
import time

import joblib
import numpy as np
import pandas as pd
import requests




MODEL_PATH = Path("models/nepal_heavy_rain_mos.joblib")
LOG_PATH = Path("logs/live_predictions.csv")
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TZ_NAME = "Asia/Kathmandu"
NPT = timezone(timedelta(hours=5, minutes=45))

API_MODELS = {
    "ecmwf": ("ecmwf_ifs025", ["precipitation", "dew_point_2m"]),
    "gfs": ("gfs_seamless", ["precipitation", "dew_point_2m", "cape"]),
    "icon": ("icon_seamless", ["precipitation", "dew_point_2m"]),
}

# Calibrated 3-tier alert scale from the untouched 2025 holdout audit
ALERT_THRESHOLDS = {
    "watch": 0.15,     # 40.0% precision
    "advisory": 0.35,  # 65.0% precision
    "warning": 0.50,   # 86.7% precision
}


def get_with_retry(url: str, params: dict, max_retries: int = 3) -> list | dict:
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = 2.0 * (attempt + 1)
                print(f"  HTTP {resp.status_code}, retrying in {wait:.0f}s...", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2.0 * (attempt + 1)
            print(f"  Request error ({exc}), retrying in {wait:.0f}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts.")


def fetch_all_cities_for_model(
    short_name: str,
    api_model: str,
    hourly_vars: list[str],
    cities: list[dict],
) -> pd.DataFrame:
    payload = get_with_retry(
        FORECAST_URL,
        {
            "latitude": ",".join(str(c["latitude"]) for c in cities),
            "longitude": ",".join(str(c["longitude"]) for c in cities),
            "hourly": ",".join(hourly_vars),
            "models": api_model,
            "past_days": 1,
            "forecast_days": 4,
            "timezone": TZ_NAME,
        },
    )
    if isinstance(payload, dict):
        payload = [payload]

    frames = []
    for city, item in zip(cities, payload):
        h = item["hourly"]
        df = pd.DataFrame({"time": pd.to_datetime(h["time"])})
        df["precip"] = pd.to_numeric(h["precipitation"], errors="coerce")
        df["dew"] = pd.to_numeric(h["dew_point_2m"], errors="coerce")
        if "cape" in hourly_vars:
            df["cape"] = pd.to_numeric(h["cape"], errors="coerce")

        df["precip_3h"] = df["precip"].rolling(3, min_periods=3).sum()
        df["date"] = df["time"].dt.floor("D")

        grp = df.groupby("date")
        out = pd.DataFrame({
            "date": grp.size().index,
            "location": city["location"],
            f"{short_name}_rain_mm": grp["precip"].sum(min_count=18).values,
            f"{short_name}_max_3h_mm": grp["precip_3h"].max().values,
            f"{short_name}_dew_mean_c": grp["dew"].mean().values,
        })
        if "cape" in df.columns:
            out[f"{short_name}_cape_max"] = grp["cape"].max().values
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def classify_alert(prob: float) -> str:
    if prob >= ALERT_THRESHOLDS["warning"]:
        return "WARNING"
    if prob >= ALERT_THRESHOLDS["advisory"]:
        return "ADVISORY"
    if prob >= ALERT_THRESHOLDS["watch"]:
        return "WATCH"
    return "Normal"


def fmt_mm(val: float) -> str:
    return f"{val:5.1f}" if pd.notna(val) else "  N/A"


def append_to_log(df: pd.DataFrame, run_dt: datetime, log_path: Path = LOG_PATH) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_date_str = run_dt.strftime("%Y-%m-%d")
    run_ts_str = run_dt.strftime("%Y-%m-%d %H:%M:%S")
    run_date_ts = pd.Timestamp(run_date_str)

    log_rows = pd.DataFrame({
        "run_date_npt": run_date_str,
        "run_timestamp_npt": run_ts_str,
        "target_date": df["date"].dt.strftime("%Y-%m-%d"),
        "lead_days": (df["date"] - run_date_ts).dt.days,
        "location": df["location"],
        "alert": df["alert"],
        "prob_50mm": df["prob_50mm"].round(4),
        "expected_mm": df["expected_mm"].round(2),
        "ecmwf_rain_mm": df["ecmwf_rain_mm"].round(2),
        "gfs_rain_mm": df["gfs_rain_mm"].round(2),
        "icon_rain_mm": df["icon_rain_mm"].round(2),
        "ens3_max_3h_mm": df["ens3_max_3h_mm"].round(2),
        "gfs_cape_max": df["gfs_cape_max"].round(1),
        "ens3_dew_mean_c": df["ens3_dew_mean_c"].round(2),
    })

    if log_path.exists():
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, log_rows], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["run_date_npt", "target_date", "location"],
            keep="last",
        )
    else:
        combined = log_rows

    combined = combined.sort_values(
        ["run_date_npt", "target_date", "location"]
    ).reset_index(drop=True)
    combined.to_csv(log_path, index=False)
    return len(combined)


def build_alert_text(df: pd.DataFrame, run_dt: datetime, min_prob: float) -> tuple[str, str]:
    alerts = df[df["prob_50mm"] >= min_prob].sort_values(
        ["prob_50mm", "date"], ascending=[False, True]
    )
    run_label = run_dt.strftime("%Y-%m-%d %H:%M NPT")

    if alerts.empty:
        subject = f"[Nepal Rain MOS] All Clear - {run_dt.strftime('%Y-%m-%d')}"
        top = df.sort_values("prob_50mm", ascending=False).iloc[0]
        body = (
            f"Nepal Heavy Rainfall MOS ({run_label})\n"
            f"Status: All 6 cities NORMAL (< {min_prob:.0%})\n"
            f"Highest 4-day signal: {top['location']} on {top['date'].strftime('%Y-%m-%d')} "
            f"({top['prob_50mm']:.1%}, Exp {top['expected_mm']:.1f} mm)"
        )
        return subject, body

    highest_tier = alerts.iloc[0]["alert"]
    subject = f"[Nepal Rain {highest_tier}] {len(alerts)} Heavy-Rain Alert(s) - {run_dt.strftime('%Y-%m-%d')}"
    lines = [
        f"Nepal Heavy Rainfall MOS ({run_label})",
        f"Active alerts (>= {min_prob:.0%} prob of >= 50 mm/day):",
        "",
    ]
    for _, r in alerts.iterrows():
        lines.append(
            f"- {r['alert']} | {r['date'].strftime('%Y-%m-%d')} | {r['location']}: "
            f"{r['prob_50mm']:.1%} (Exp {r['expected_mm']:.1f} mm; "
            f"ECMWF {r['ecmwf_rain_mm']:.0f} / GFS {r['gfs_rain_mm']:.0f} / ICON {r['icon_rain_mm']:.0f} mm)"
        )
    return subject, "\n".join(lines)


def send_telegram(message: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False

    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message},
        timeout=20,
    )
    resp.raise_for_status()
    print("Telegram alert sent.", flush=True)
    return True


def send_email(subject: str, body: str) -> bool:
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_APP_PASSWORD")
    email_to = os.getenv("ALERT_EMAIL_TO")
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    if not smtp_user or not smtp_pass or not email_to:
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = email_to
    msg.set_content(body)

    with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20) as server:
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
    print(f"Email alert sent to {email_to}.", flush=True)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live Nepal heavy-rainfall MOS forecast.")
    parser.add_argument("--no-log", action="store_true", help="Skip appending to logs/live_predictions.csv")
    parser.add_argument(
        "--force-notify",
        action="store_true",
        help="Send Telegram/Email summary even when all cities are Normal",
    )
    parser.add_argument(
        "--min-alert-prob",
        type=float,
        default=ALERT_THRESHOLDS["watch"],
        help="Minimum probability threshold to trigger alerts (default: 0.15)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run `python src/train_production_model.py` first."
        )

    run_dt = datetime.now(NPT)
    art = joblib.load(MODEL_PATH)
    models = art["models"]
    ratio_map = art["ratio_map"]
    cities = art["cities"]

    merged = None
    for idx, short_name in enumerate(models, start=1):
        api_model, hourly_vars = API_MODELS[short_name]
        print(f"[{idx}/{len(models)}] Fetching {short_name.upper()} ({api_model}) for {len(cities)} cities...", flush=True)
        model_df = fetch_all_cities_for_model(short_name, api_model, hourly_vars, cities)
        merged = model_df if merged is None else merged.merge(model_df, on=["date", "location"], how="inner")

    df = merged.sort_values(["location", "date"]).reset_index(drop=True)

    # Feature engineering (identical to src/train_production_model.py)
    df["ens3_raw_mm"] = df[[f"{m}_rain_mm" for m in models]].mean(axis=1)
    df["ens3_max_3h_mm"] = df[[f"{m}_max_3h_mm" for m in models]].mean(axis=1)
    df["ens3_dew_mean_c"] = df[[f"{m}_dew_mean_c" for m in models]].mean(axis=1)

    day_sum = df.groupby("date")["ens3_raw_mm"].transform("sum")
    day_cnt = df.groupby("date")["ens3_raw_mm"].transform("count")
    df["ens3_other_cities_mm"] = (day_sum - df["ens3_raw_mm"]) / np.maximum(day_cnt - 1, 1)
    df["ens3_prev_day_mm"] = df.groupby("location")["ens3_raw_mm"].shift(1).fillna(0.0)

    # Drop the past_days=1 anchor date so only today + upcoming days are logged/displayed
    first_date = df["date"].min()
    df = df[df["date"] > first_date].reset_index(drop=True)

    bias_cols = [
        (df[f"{m}_rain_mm"] * df["location"].map(ratio_map[m]["per_city"]).fillna(ratio_map[m]["global"])).values
        for m in models
    ]
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

    df["prob_50mm"] = art["clf_50mm"].predict_proba(X_clf)[:, 1]
    raw_expected = np.clip(art["reg_mm"].predict(X_reg), 0.0, None)
    df["expected_mm"] = np.where(df["ens3_raw_mm"] < 0.1, 0.0, raw_expected)
    df["alert"] = [classify_alert(p) for p in df["prob_50mm"]]

    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    view = df[[
        "date_str", "location", "alert", "prob_50mm", "expected_mm",
        "ecmwf_rain_mm", "gfs_rain_mm", "icon_rain_mm", "ens3_max_3h_mm",
    ]].sort_values(["date_str", "prob_50mm"], ascending=[True, False])

    view = view.rename(columns={
        "date_str": "Date",
        "location": "City",
        "alert": "Alert",
        "prob_50mm": "P(>=50mm)",
        "expected_mm": "Exp_mm",
        "ecmwf_rain_mm": "ECMWF",
        "gfs_rain_mm": "GFS",
        "icon_rain_mm": "ICON",
        "ens3_max_3h_mm": "Peak3h",
    })
    view["P(>=50mm)"] = view["P(>=50mm)"].map(lambda x: f"{x:6.1%}")
    for col in ["Exp_mm", "ECMWF", "GFS", "ICON", "Peak3h"]:
        view[col] = view[col].map(fmt_mm)

    print(f"\n=== NEPAL HEAVY RAINFALL MOS FORECAST ({run_dt.strftime('%Y-%m-%d %H:%M NPT')}) ===")
    print("Alert tiers: WATCH >= 15% | ADVISORY >= 35% | WARNING >= 50%\n")
    print(view.to_string(index=False))

    if not args.no_log:
        total_rows = append_to_log(df, run_dt)
        print(f"\nLogged to {LOG_PATH} ({total_rows} total rows).")

    has_alerts = (df["prob_50mm"] >= args.min_alert_prob).any()
    if has_alerts or args.force_notify:
        subject, body = build_alert_text(df, run_dt, args.min_alert_prob)
        sent_tg = send_telegram(body)
        sent_email = send_email(subject, body)
        if not (sent_tg or sent_email):
            print("Alert condition met, but no TELEGRAM_* or SMTP_* environment variables are configured.")


if __name__ == "__main__":
    main()