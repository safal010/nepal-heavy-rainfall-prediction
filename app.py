from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.nepal_regions import BELT_ORDER, PROVINCE_ORDER, get_all_locations

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

ALERT_COLORS = {
    "Normal": "#22c55e",
    "WATCH": "#3b82f6",
    "ADVISORY": "#f59e0b",
    "WARNING": "#ef4444",
}

ALERT_SIZES = {
    "Normal": 12000,
    "WATCH": 18000,
    "ADVISORY": 24000,
    "WARNING": 32000,
}

st.set_page_config(
    page_title="Nepal Heavy Rainfall Prediction System",
    page_icon="🌧️",
    layout="wide",
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
    batch_size: int = 25,
) -> pd.DataFrame:
    frames = []
    for start in range(0, len(cities), batch_size):
        batch = cities[start : start + batch_size]
        payload = get_with_retry(
            FORECAST_URL,
            {
                "latitude": ",".join(str(c["latitude"]) for c in batch),
                "longitude": ",".join(str(c["longitude"]) for c in batch),
                "hourly": ",".join(hourly_vars),
                "models": api_model,
                "past_days": 1,
                "forecast_days": 4,
                "timezone": TZ_NAME,
            },
        )
        if isinstance(payload, dict):
            payload = [payload]

        for city, item in zip(batch, payload):
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


@st.cache_data(ttl=1800, show_spinner="Fetching live ECMWF, GFS, and ICON forecasts across all 7 provinces...")
def get_live_mos_forecasts(model_path_str: str) -> tuple[pd.DataFrame, list[dict], str]:
    art = load_artifact(model_path_str)
    models = art["models"]
    ratio_map = art["ratio_map"]
    locations = get_all_locations(art["cities"])
    meta_df = pd.DataFrame(locations)

    merged = None
    for short_name in models:
        api_model, hourly_vars = API_MODELS[short_name]
        model_df = fetch_all_cities_for_model(short_name, api_model, hourly_vars, locations)
        merged = (
            model_df
            if merged is None
            else merged.merge(model_df, on=["date", "location"], how="inner")
        )

    df = merged.merge(meta_df, on="location", how="left")
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    # Feature engineering (identical to src/train_production_model.py & src/predict_live.py)
    df["ens3_raw_mm"] = df[[f"{m}_rain_mm" for m in models]].mean(axis=1)
    df["ens3_max_3h_mm"] = df[[f"{m}_max_3h_mm" for m in models]].mean(axis=1)
    df["ens3_dew_mean_c"] = df[[f"{m}_dew_mean_c" for m in models]].mean(axis=1)

    day_sum = df.groupby("date")["ens3_raw_mm"].transform("sum")
    day_cnt = df.groupby("date")["ens3_raw_mm"].transform("count")
    df["ens3_other_cities_mm"] = (day_sum - df["ens3_raw_mm"]) / np.maximum(day_cnt - 1, 1)
    df["ens3_prev_day_mm"] = df.groupby("location")["ens3_raw_mm"].shift(1).fillna(0.0)

    # Drop the past_days=1 anchor date after computing lag features
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
    return df, locations, updated_at


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
    st.title("🌧️ Nepal Heavy Rainfall Prediction System")
    st.write(
        "Live 3-model ensemble MOS risk estimates across all **7 Provinces** and "
        "**3 Ecological Regions (Terai, Hill, Mountain)** of Nepal."
    )
    st.info(
        "Educational prototype only. This is not an official weather warning. "
        "Forecasts and model predictions may be inaccurate."
    )

    try:
        df, locations, updated_at = get_live_mos_forecasts(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Unable to process the forecast: {exc}")
        return

    available_dates = sorted(df["date_str"].unique().tolist())

    # Top controls: Province, Ecological Belt, Location, Date, Refresh
    f1, f2, f3, f4, f5 = st.columns([2, 1.5, 2, 1.8, 1], vertical_alignment="bottom")

    provinces = ["All Provinces"] + [p for p in PROVINCE_ORDER if p in set(df["province"])]
    with f1:
        selected_province = st.selectbox("Province", provinces)

    belts = ["All Regions"] + [b for b in BELT_ORDER if b in set(df["belt"])]
    with f2:
        selected_belt = st.selectbox("Ecological Region", belts)

    filtered_locs = [
        loc
        for loc in locations
        if (selected_province == "All Provinces" or loc["province"] == selected_province)
        and (selected_belt == "All Regions" or loc["belt"] == selected_belt)
    ]
    if not filtered_locs:
        st.warning(f"No stations in {selected_province} belong to the {selected_belt} belt. Showing all stations in {selected_province}.")
        filtered_locs = [
            loc
            for loc in locations
            if selected_province == "All Provinces" or loc["province"] == selected_province
        ]

    loc_names = [loc["location"] for loc in filtered_locs]
    default_idx = loc_names.index("Kathmandu") if "Kathmandu" in loc_names else 0

    with f3:
        selected_city = st.selectbox("Location", loc_names, index=default_idx)
    with f4:
        selected_date = st.selectbox("Forecast Date", available_dates)
    with f5:
        if st.button("Refresh", width="stretch"):
            get_live_mos_forecasts.clear()
            st.rerun()

    loc_meta = next(loc for loc in filtered_locs if loc["location"] == selected_city)
    st.caption(
        f"{loc_meta['province']} Province  •  {loc_meta['belt']} Region  •  "
        f"Coordinates: {loc_meta['latitude']:.4f}, {loc_meta['longitude']:.4f}  •  "
        f"{loc_meta['calibration']}  •  Updated: {updated_at}"
    )

    city_df = df[df["location"] == selected_city].sort_values("date").reset_index(drop=True)
    row = city_df[city_df["date_str"] == selected_date].iloc[0]

    left_col, right_col = st.columns([1.25, 1], gap="large")

    with left_col:
        st.subheader(f"Selected day's forecast — {selected_city} ({selected_date})")
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
                "ecmwf_rain_mm": "ECMWF",
                "gfs_rain_mm": "GFS",
                "icon_rain_mm": "ICON",
                "ens3_max_3h_mm": "Peak 3h",
            }
        )
        city_table["P(>=50mm)"] = city_table["P(>=50mm)"].map(lambda x: f"{x:.1%}")
        for col in ["Exp (mm)", "ECMWF", "GFS", "ICON", "Peak 3h"]:
            city_table[col] = city_table[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        st.dataframe(city_table, hide_index=True, width="stretch")

    with right_col:
        st.subheader(f"Nepal Regional Risk Map ({selected_date})")
        map_df = df[df["date_str"] == selected_date].copy()
        if selected_province != "All Provinces":
            map_df = map_df[map_df["province"] == selected_province]
        if selected_belt != "All Regions":
            map_df = map_df[map_df["belt"] == selected_belt]

        map_df["color"] = map_df["alert"].map(ALERT_COLORS).fillna("#22c55e")
        map_df["size"] = map_df["alert"].map(ALERT_SIZES).fillna(12000)
        st.map(
            map_df,
            latitude="latitude",
            longitude="longitude",
            color="color",
            size="size",
        )
        st.caption("Marker color: Green = Normal • Blue = WATCH • Orange = ADVISORY • Red = WARNING")

    st.divider()
    tab_prov, tab_all = st.tabs([f"Province Summary ({selected_date})", f"All {len(locations)} Nepal Regions ({selected_date})"])

    day_df = df[df["date_str"] == selected_date].copy()

    with tab_prov:
        prov_rows = []
        for prov in PROVINCE_ORDER:
            sub = day_df[day_df["province"] == prov]
            if sub.empty:
                continue
            worst = sub.sort_values(["prob_50mm", "expected_mm"], ascending=False).iloc[0]
            prov_rows.append(
                {
                    "Province": prov,
                    "Highest Risk City": f"{worst['location']} ({worst['belt']})",
                    "Peak Alert": worst["alert"],
                    "Max P(>=50mm)": f"{worst['prob_50mm']:.1%}",
                    "Max Exp (mm)": f"{worst['expected_mm']:.1f}",
                    "Stations Monitored": len(sub),
                }
            )
        st.dataframe(pd.DataFrame(prov_rows), hide_index=True, width="stretch")

    with tab_all:
        all_view = (
            day_df.sort_values(["prob_50mm", "expected_mm"], ascending=False)[
                [
                    "province",
                    "belt",
                    "location",
                    "alert",
                    "prob_50mm",
                    "expected_mm",
                    "ecmwf_rain_mm",
                    "gfs_rain_mm",
                    "icon_rain_mm",
                    "ens3_max_3h_mm",
                    "calibration",
                ]
            ]
            .rename(
                columns={
                    "province": "Province",
                    "belt": "Region",
                    "location": "Location",
                    "alert": "Alert",
                    "prob_50mm": "P(>=50mm)",
                    "expected_mm": "Exp (mm)",
                    "ecmwf_rain_mm": "ECMWF",
                    "gfs_rain_mm": "GFS",
                    "icon_rain_mm": "ICON",
                    "ens3_max_3h_mm": "Peak 3h",
                    "calibration": "Calibration",
                }
            )
            .copy()
        )
        all_view["P(>=50mm)"] = all_view["P(>=50mm)"].map(lambda x: f"{x:.1%}")
        for col in ["Exp (mm)", "ECMWF", "GFS", "ICON", "Peak 3h"]:
            all_view[col] = all_view[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        st.dataframe(all_view, hide_index=True, width="stretch")

    st.divider()
    st.caption(
        "Educational project | Forecast source: Open-Meteo (ECMWF IFS, NOAA GFS, DWD ICON)"
    )


if __name__ == "__main__":
    main()