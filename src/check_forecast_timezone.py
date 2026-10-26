
import requests

url = "https://single-runs-api.open-meteo.com/v1/forecast"

for timezone in ["UTC", "Asia/Kathmandu"]:
    params = {
        "latitude": 27.7172,
        "longitude": 85.3240,
        "hourly": "precipitation",
        "timezone": timezone,
        "models": "ecmwf_ifs",
        "run": "2025-02-28T00:00",
        "forecast_hours": 72,
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    hourly = response.json()["hourly"]

    total = sum(
        value
        for timestamp, value in zip(
            hourly["time"], hourly["precipitation"]
        )
        if timestamp.startswith("2025-03-01")
        and value is not None
    )

    print(f"{timezone}: {total:.2f} mm")
