
import requests
import pandas as pd

url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

params = {
    "latitude": 27.7172,
    "longitude": 85.3240,
    "start_date": "2025-06-01",
    "end_date": "2025-06-07",
    "daily": [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "precipitation_probability_max",
        "wind_speed_10m_max",
    ],
    "timezone": "Asia/Kathmandu",
}

print("Requesting historical forecast data for Kathmandu...")

try:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "daily" not in data:
        print("Unexpected API response:")
        print(data)
    else:
        daily = pd.DataFrame(data["daily"])

        print("\n--- FORECAST DATA SAMPLE ---")
        print(daily.to_string(index=False))

        print("\n--- UNITS ---")
        print(data.get("daily_units", {}))

except requests.RequestException as error:
    print("API request failed:", error)
except ValueError as error:
    print("Could not read the API response:", error)
