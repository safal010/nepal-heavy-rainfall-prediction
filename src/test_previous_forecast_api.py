
import requests
import pandas as pd

url = "https://previous-runs-api.open-meteo.com/v1/forecast"

params = {
    "latitude": 27.7172,
    "longitude": 85.3240,
    "start_date": "2025-06-01",
    "end_date": "2025-06-03",
    "hourly": [
        "temperature_2m_previous_day1",
        "relative_humidity_2m_previous_day1",
        "precipitation_previous_day1",
        "wind_speed_10m_previous_day1",
    ],
    "timezone": "Asia/Kathmandu",
}

print("Testing one-day-ahead historical forecasts for Kathmandu...")

try:
    response = requests.get(url, params=params, timeout=60)

    if not response.ok:
        print("HTTP status:", response.status_code)
        print("API error details:", response.text)
    else:
        data = response.json()

        if "hourly" not in data:
            print("Unexpected API response:")
            print(data)
        else:
            hourly = pd.DataFrame(data["hourly"])

            print("\n--- FIRST 12 HOURLY RECORDS ---")
            print(hourly.head(12).to_string(index=False))

            print("\n--- UNITS ---")
            print(data.get("hourly_units", {}))

            print("\n--- MISSING VALUES ---")
            print(hourly.isna().sum())

except requests.RequestException as error:
    print("API request failed:", error)
except ValueError as error:
    print("Could not read the API response:", error)
