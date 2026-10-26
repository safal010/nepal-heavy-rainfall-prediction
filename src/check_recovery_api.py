
import requests
import pandas as pd

API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

locations = {
    "Biratnagar": (26.4525, 87.2718),
    "Janakpur": (26.7288, 85.9263),
    "Jumla": (29.2747, 82.1838),
    "Kathmandu": (27.7172, 85.3240),
    "Nepalgunj": (28.0500, 81.6167),
    "Pokhara": (28.2096, 83.9856),
}

params_base = {
    "start_date": "2025-08-06",
    "end_date": "2025-08-10",
    "hourly": (
        "temperature_2m,relative_humidity_2m,"
        "precipitation,wind_speed_10m"
    ),
    "timezone": "Asia/Kathmandu",
    "models": "ecmwf_ifs025",
}

for location, (latitude, longitude) in locations.items():
    params = {
        **params_base,
        "latitude": latitude,
        "longitude": longitude,
    }

    try:
        response = requests.get(API_URL, params=params, timeout=60)
        print(f"\n{location}: HTTP {response.status_code}")

        if response.status_code != 200:
            print(response.text[:300])
            continue

        hourly = response.json().get("hourly", {})
        times = hourly.get("time", [])
        print("Hourly records:", len(times))

        for variable in [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
        ]:
            values = hourly.get(variable, [])
            missing = sum(v is None for v in values)
            print(f"  {variable}: {len(values)} values, {missing} missing")

    except requests.RequestException as error:
        print("Request failed:", error)
