
import requests
import time

API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"

LOCATIONS = {
    "Kathmandu": (27.7172, 85.3240),
    "Pokhara": (28.2096, 83.9856),
}

TARGET_DATES = [
    "2025-06-15",
    "2025-09-02",
    "2025-11-15",
]

for location, (latitude, longitude) in LOCATIONS.items():
    for target_date in TARGET_DATES:
        run_date = (
            __import__("datetime").date.fromisoformat(target_date)
            - __import__("datetime").timedelta(days=1)
        ).isoformat()

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "precipitation",
            "timezone": "Asia/Kathmandu",
            "models": "ecmwf_ifs",
            "run": f"{run_date}T00:00",
            "forecast_hours": 72,
        }

        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=60,
            )

            if response.status_code != 200:
                print(
                    f"{location} | {target_date} | "
                    f"HTTP {response.status_code} | "
                    f"{response.text[:200]}"
                )
                continue

            data = response.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            values = hourly.get("precipitation", [])

            target_values = [
                value
                for timestamp, value in zip(times, values)
                if timestamp[:10] == target_date
                and value is not None
            ]

            status = (
                "AVAILABLE (24 hours)"
                if len(target_values) == 24
                else f"INCOMPLETE ({len(target_values)} hours)"
            )

            print(
                f"{location} | {target_date} | "
                f"run={run_date}T00:00 UTC | {status}"
            )

        except requests.RequestException as error:
            print(f"{location} | {target_date} | ERROR: {error}")

        time.sleep(1)
