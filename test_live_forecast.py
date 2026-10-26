
import requests

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": 27.7172,
    "longitude": 85.3240,
    "daily": [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_mean",
    ],
    "forecast_days": 3,
    "timezone": "Asia/Kathmandu",
}

try:
    response = requests.get(url, params=params, timeout=30)
    print("Status code:", response.status_code)
    response.raise_for_status()

    data = response.json()
    print("Location:", data.get("latitude"), data.get("longitude"))
    print("Timezone:", data.get("timezone"))
    print("Daily forecast:")
    print(data.get("daily"))

except requests.RequestException as error:
    print("Forecast request failed:", error)
