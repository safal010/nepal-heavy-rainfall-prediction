
import streamlit as st
import joblib
import pandas as pd
import requests
from pathlib import Path

st.set_page_config(
    page_title="Heavy Rainfall Prediction - Nepal",
    page_icon="🌧️",
    layout="wide"
)

MODEL_PATH = Path("models/fixed_run_heavy_rain_model.pkl")
FEATURES_PATH = Path("models/fixed_run_model_features.pkl")

LOCATIONS = {
    "Kathmandu": (27.7172, 85.3240),
    "Pokhara": (28.2096, 83.9856),
    "Biratnagar": (26.4525, 87.2718),
    "Janakpur": (26.7288, 85.9263),
    "Jumla": (29.2747, 82.1838),
    "Nepalgunj": (28.0500, 81.6167),
}

API_URL = "https://api.open-meteo.com/v1/forecast"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


@st.cache_data(ttl=1800)
def get_forecast(latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_mean",
        ],
        "forecast_days": 3,
        "timezone": "Asia/Kathmandu",
    }

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    daily = data.get("daily", {})

    required = [
        "time",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_mean",
    ]

    if any(key not in daily for key in required):
        raise ValueError("The forecast API did not return all required fields.")

    lengths = [len(daily[key]) for key in required]
    if len(set(lengths)) != 1 or not lengths or lengths[0] == 0:
        raise ValueError("The forecast API returned incomplete daily data.")

    if any(
        any(value is None for value in daily[key])
        for key in required[1:]
    ):
        raise ValueError("Some forecast values are missing.")

    return pd.DataFrame({
        "date": daily["time"],
        "temperature_mean_c": daily["temperature_2m_mean"],
        "humidity_mean_pct": daily["relative_humidity_2m_mean"],
        "forecast_rainfall_mm": daily["precipitation_sum"],
        "wind_speed_mean_kmh": daily["wind_speed_10m_mean"],
    })


model, features = load_model()

st.title("🌧️ Heavy Rainfall Prediction System")
st.write("Live forecast-based rainfall risk estimates for selected locations in Nepal.")

st.warning(
    "Educational prototype only. This is not an official weather warning. "
    "Forecasts and model predictions may be inaccurate."
)

location = st.selectbox("Select a location", list(LOCATIONS.keys()))
latitude, longitude = LOCATIONS[location]

st.caption(f"Coordinates: {latitude}, {longitude}")

try:
    forecast = get_forecast(latitude, longitude)

    st.subheader(f"Three-day forecast for {location}")

    display_forecast = forecast.rename(columns={
        "date": "Date",
        "temperature_mean_c": "Mean temperature (°C)",
        "humidity_mean_pct": "Mean humidity (%)",
        "forecast_rainfall_mm": "Forecast rainfall (mm)",
        "wind_speed_mean_kmh": "Mean wind speed (km/h)",
    })

    st.dataframe(display_forecast, hide_index=True, use_container_width=True)

    selected_date = st.selectbox(
        "Choose a forecast date for risk estimation",
        forecast["date"].tolist()
    )

    selected_row = forecast.loc[forecast["date"] == selected_date].iloc[0]

    st.subheader("Selected day's forecast")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", f"{selected_row['temperature_mean_c']:.1f} °C")
    col2.metric("Humidity", f"{selected_row['humidity_mean_pct']:.0f}%")
    col3.metric("Rainfall", f"{selected_row['forecast_rainfall_mm']:.1f} mm")
    col4.metric("Wind speed", f"{selected_row['wind_speed_mean_kmh']:.1f} km/h")

    if st.button("Predict Heavy-Rainfall Risk", type="primary"):
        input_data = pd.DataFrame([{
            "latitude": latitude,
            "longitude": longitude,
            "temperature_mean_c": selected_row["temperature_mean_c"],
            "humidity_mean_pct": selected_row["humidity_mean_pct"],
            "forecast_rainfall_mm": selected_row["forecast_rainfall_mm"],
            "wind_speed_mean_kmh": selected_row["wind_speed_mean_kmh"],
        }])

        input_data = input_data[features]

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        heavy_index = list(model.classes_).index(1)
        probability = float(probabilities[heavy_index]) * 100

        st.subheader(f"Risk estimate for {location} on {selected_date}")
        st.metric("Model-estimated heavy-rain probability", f"{probability:.1f}%")

        if prediction == 1:
            st.error("The model classifies this forecast as a heavy-rain event.")
        else:
            st.success("The model does not classify this forecast as a heavy-rain event.")

        st.caption(
            "Heavy rainfall is defined as at least 50 mm/day in the training labels. "
            "The probability is a model estimate, not a calibrated official warning."
        )

except (requests.RequestException, ValueError, KeyError, IndexError) as error:
    st.error(f"Unable to load or process the forecast: {error}")

st.divider()
st.caption("Educational project | Forecast source: Open-Meteo.")
