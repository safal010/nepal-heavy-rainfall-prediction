
import requests

url = "https://previous-runs-api.open-meteo.com/v1/forecast"

models = ["ecmwf_ifs", "ecmwf_ifs025"]

params_base = {
    "latitude": 27.7172,
    "longitude": 85.3240,
    "hourly": "precipitation_previous_day1",
    "start_date": "2025-08-06",
    "end_date": "2025-08-06",
    "timezone": "UTC",
}

for model in models:
    params = {**params_base, "models": model}

    print(f"\nTesting model: {model}")

    try:
        response = requests.get(url, params=params, timeout=60)
        print("HTTP status:", response.status_code)

        if not response.ok:
            print("API error:", response.text)
            continue

        data = response.json()
        hourly = data.get("hourly", {})
        rainfall = hourly.get("precipitation_previous_day1", [])

        valid = [value for value in rainfall if value is not None]

        print("Hourly records:", len(rainfall))
        print("Valid values:", len(valid))

        if len(valid) == 24:
            print(f"Daily forecast rainfall: {sum(valid):.2f} mm")
        else:
            print("This model did not return 24 valid hourly values.")

    except requests.RequestException as error:
        print("Request failed:", error)
