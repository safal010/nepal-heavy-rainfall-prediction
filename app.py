from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

MODEL_PATH = Path("models/nepal_heavy_rain_mos.joblib")
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TZ_NAME = "Asia/Kathmandu"
NPT = timezone(timedelta(hours=5, minutes=45))

API_MODELS = {
    "ecmwf": ("ecmwf_ifs025", ["precipitation", "dew_point_2m"]),
    "gfs": (
        "gfs_seamless",
        [
            "precipitation",
            "dew_point_2m",
            "cape",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
        ],
    ),
    "icon": ("icon_seamless", ["precipitation", "dew_point_2m"]),
}

ALERT_THRESHOLDS = {
    "watch": 0.15,
    "advisory": 0.35,
    "warning": 0.50,
}

st.set_page_config(
    page_title="Heavy Rainfall Prediction System",
    page_icon="🌧️",
    layout="centered",
)


def get_with_retry(url: str, params: dict, max_retries: int = 3) -> list | dict:
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
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
        if "temperature_2m" in hourly_vars:
            df["temp"] = pd.to_numeric(h["temperature_2m"], errors="coerce")
        if "relative_humidity_2m" in hourly_vars:
            df["rh"] = pd.to_numeric(h["relative_humidity_2m"], errors="coerce")
        if "wind_speed_10m" in hourly_vars:
            df["wind"] = pd.to_numeric(h["wind_speed_10m"], errors="coerce")

        df["precip_3h"] = df["precip"].rolling(3, min_periods=3).sum()
        df["date"] = df["time"].dt.floor("D")

        grp = df.groupby("date")
        out = pd.DataFrame(
            {
                "date": grp.size().index,
                "location": city["location"],
                f"{short_name}_rain_mm": grp["precip"].sum(min_count=18).values,
                f"{short_name}_max_3h_mm": grp["precip_3h"].max().values,
                f"{short_name}_dew_mean_c": grp["dew"].mean().values,
            }
        )
        if "cape" in df.columns:
            out[f"{short_name}_cape_max"] = grp["cape"].max().values
        if "temp" in df.columns:
            out["temp_mean_c"] = grp["temp"].mean().values
        if "rh" in df.columns:
            out["humidity_mean_pct"] = grp["rh"].mean().values
        if "wind" in df.columns:
            out["wind_mean_kmh"] = grp["wind"].mean().values

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


@st.cache_resource
def load_artifact(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at `{path}`. Run `python src/train_production_model.py` first."
        )
    return joblib.load(path)


@st.cache_data(ttl=1800, show_spinner="Fetching live ECMWF, GFS, and ICON forecasts...")
def get_live_mos_forecasts(model_path_str: str) -> tuple[pd.DataFrame, str]:
    art = load_artifact(model_path_str)
    models = art["models"]
    ratio_map = art["ratio_map"]
    cities = art["cities"]

    merged = None
    for short_name in models:
        api_model, hourly_vars = API_MODELS[short_name]
        model_df = fetch_all_cities_for_model(short_name, api_model, hourly_vars, cities)
        merged = (
            model_df
            if merged is None
            else merged.merge(model_df, on=["date", "location"], how="inner")
        )

    df = merged.sort_values(["location", "date"]).reset_index(drop=True)

    # Feature engineering (identical to src/train_production_model.py & src/predict_live.py)
    df["ens3_raw_mm"] = df[[f"{m}_rain_mm" for m in models]].mean(axis=1)
    df["ens3_max_3h_mm"] = df[[f"{m}_max_3h_mm" for m in models]].mean(axis=1)
    df["ens3_dew_mean_c"] = df[[f"{m}_dew_mean_c" for m in models]].mean(axis=1)

    day_sum = df.groupby("date")["ens3_raw_mm"].transform("sum")
    day_cnt = df.groupby("date")["ens3_raw_mm"].transform("count")
    df["ens3_other_cities_mm"] = (day_sum - df["ens3_raw_mm"]) / np.maximum(day_cnt - 1, 1)
    df["ens3_prev_day_mm"] = df.groupby("location")["ens3_raw_mm"].shift(1).fillna(0.0)

    # Drop the past_days=1 anchor row after computing lag features
    first_date = df["date"].min()
    df = df[df["date"] > first_date].reset_index(drop=True)

    bias_cols = [
        (
            df[f"{m}_rain_mm"]
            * df["location"].map(ratio_map[m]["per_city"]).fillna(ratio_map[m]["global"])
        ).values
        for m in models
    ]
    X_reg = np.column_stack(
        bias_cols
        + [
            df["ens3_other_cities_mm"].values,
            df["ens3_prev_day_mm"].values,
        ]
    )
    X_clf = np.column_stack(
        [
            X_reg,
            df["ens3_max_3h_mm"].values,
            df["gfs_cape_max"].values,
            df["ens3_dew_mean_c"].values,
        ]
    )

    df["prob_50mm"] = art["clf_50mm"].predict_proba(X_clf)[:, 1]
    raw_expected = np.clip(art["reg_mm"].predict(X_reg), 0.0, None)
    df["expected_mm"] = np.where(df["ens3_raw_mm"] < 0.1, 0.0, raw_expected)
    df["alert"] = [classify_alert(p) for p in df["prob_50mm"]]
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    updated_at = datetime.now(NPT).strftime("%Y-%m-%d %H:%M NPT")
    return df, updated_at


def render_alert_banner(alert: str, prob: float, expected_mm: float, city: str, date_str: str) -> None:
    msg = (
        f"**{alert}** — {city} ({date_str}): "
        f"**{prob:.1%}** probability of heavy rainfall (≥ 50 mm/day) | "
        f"Calibrated expected rainfall: **{expected_mm:.1f} mm**"
    )
    if alert == "WARNING":
        st.error(msg)
    elif alert == "ADVISORY":
        st.warning(msg)
    elif alert == "WATCH":
        st.info(msg)
    else:
        st.success(msg)


def main() -> None:
    st.title("🌧️ Heavy Rainfall Prediction System")
    st.write("Live forecast-based rainfall risk estimates for selected locations in Nepal.")
    st.info(
        "Educational prototype only. This is not an official weather warning. "
        "Forecasts and model predictions may be inaccurate."
    )

    try:
        art = load_artifact(str(MODEL_PATH))
        df, updated_at = get_live_mos_forecasts(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Unable to process the forecast: {exc}")
        return

    city_lookup = {c["location"]: c for c in art["cities"]}
    city_names = list(city_lookup.keys())
    default_idx = city_names.index("Kathmandu") if "Kathmandu" in city_names else 0

    col_sel, col_btn = st.columns([4, 1], vertical_alignment="bottom")
    with col_sel:
        selected_city = st.selectbox("Select a location", city_names, index=default_idx)
    with col_btn:
        if st.button("Refresh", width="stretch"):
            get_live_mos_forecasts.clear()
            st.rerun()

    city_meta = city_lookup[selected_city]
    st.caption(
        f"Coordinates: {city_meta['latitude']}, {city_meta['longitude']}  •  "
        f"Last updated: {updated_at}"
    )

    city_df = df[df["location"] == selected_city].sort_values("date").reset_index(drop=True)
    available_dates = city_df["date_str"].tolist()

    st.subheader(f"Four-day forecast for {selected_city}")
    selected_date = st.selectbox("Choose a forecast date for risk estimation", available_dates)

    row = city_df[city_df["date_str"] == selected_date].iloc[0]

    st.markdown("#### Selected day's forecast")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Temperature",
        f"{row['temp_mean_c']:.1f} °C" if pd.notna(row.get("temp_mean_c")) else "N/A",
    )
    c2.metric(
        "Humidity",
        f"{row['humidity_mean_pct']:.0f}%" if pd.notna(row.get("humidity_mean_pct")) else "N/A",
    )
    c3.metric("Rainfall (MOS)", f"{row['expected_mm']:.1f} mm")
    c4.metric(
        "Wind speed",
        f"{row['wind_mean_kmh']:.1f} km/h" if pd.notna(row.get("wind_mean_kmh")) else "N/A",
    )

    st.markdown("#### Heavy rainfall risk assessment")
    render_alert_banner(
        row["alert"],
        float(row["prob_50mm"]),
        float(row["expected_mm"]),
        selected_city,
        selected_date,
    )
    st.caption("Alert tiers: WATCH ≥ 15%  |  ADVISORY ≥ 35%  |  WARNING ≥ 50%")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ECMWF", f"{row['ecmwf_rain_mm']:.1f} mm")
    m2.metric("GFS", f"{row['gfs_rain_mm']:.1f} mm")
    m3.metric("ICON", f"{row['icon_rain_mm']:.1f} mm")
    m4.metric("Peak 3h Rain", f"{row['ens3_max_3h_mm']:.1f} mm")

    st.markdown(f"#### 4-day outlook — {selected_city}")
    city_table = city_df[
        [
            "date_str",
            "alert",
            "prob_50mm",
            "expected_mm",
            "ecmwf_rain_mm",
            "gfs_rain_mm",
            "icon_rain_mm",
            "ens3_max_3h_mm",
        ]
    ].rename(
        columns={
            "date_str": "Date",
            "alert": "Alert",
            "prob_50mm": "P(>=50mm)",
            "expected_mm": "Exp (mm)",
            "ecmwf_rain_mm": "ECMWF (mm)",
            "gfs_rain_mm": "GFS (mm)",
            "icon_rain_mm": "ICON (mm)",
            "ens3_max_3h_mm": "Peak 3h (mm)",
        }
    )
    city_table["P(>=50mm)"] = city_table["P(>=50mm)"].map(lambda x: f"{x:.1%}")
    for col in ["Exp (mm)", "ECMWF (mm)", "GFS (mm)", "ICON (mm)", "Peak 3h (mm)"]:
        city_table[col] = city_table[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")

    st.dataframe(city_table, hide_index=True, width="stretch")

    with st.expander("View all 6 Nepal cities"):
        all_table = (
            df.sort_values(["date_str", "prob_50mm"], ascending=[True, False])[
                ["date_str", "location", "alert", "prob_50mm", "expected_mm", "ens3_max_3h_mm"]
            ]
            .rename(
                columns={
                    "date_str": "Date",
                    "location": "City",
                    "alert": "Alert",
                    "prob_50mm": "P(>=50mm)",
                    "expected_mm": "Exp (mm)",
                    "ens3_max_3h_mm": "Peak 3h (mm)",
                }
            )
            .copy()
        )
        all_table["P(>=50mm)"] = all_table["P(>=50mm)"].map(lambda x: f"{x:.1%}")
        all_table["Exp (mm)"] = all_table["Exp (mm)"].map(lambda x: f"{x:.1f}")
        all_table["Peak 3h (mm)"] = all_table["Peak 3h (mm)"].map(lambda x: f"{x:.1f}")
        st.dataframe(all_table, hide_index=True, width="stretch")

    st.divider()
    st.caption(
        "Educational project | Forecast source: Open-Meteo (ECMWF IFS, NOAA GFS, DWD ICON)"
    )


if __name__ == "__main__":
    main()