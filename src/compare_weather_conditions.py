
import pandas as pd

# Load prepared weather data
df = pd.read_csv("data/prepared_weather.csv")

# Separate days by whether tomorrow has heavy rainfall
heavy = df[df["heavy_rain_tomorrow"] == 1]
non_heavy = df[df["heavy_rain_tomorrow"] == 0]

# Weather features to compare
features = [
    "T2M",
    "RH2M",
    "PS",
    "WS10M",
    "PRECTOTCORR",
    "rainfall_1_day_ago_mm",
    "rainfall_2_day_ago_mm",
    "rainfall_3_day_ago_mm",
]

print("\n--- NUMBER OF CASES ---")
print("Heavy-rainfall cases:", len(heavy))
print("Non-heavy-rainfall cases:", len(non_heavy))

print("\n--- AVERAGE WEATHER CONDITIONS ---")

comparison = pd.DataFrame({
    "Heavy_rain_tomorrow": heavy[features].mean(),
    "No_heavy_rain_tomorrow": non_heavy[features].mean(),
})

comparison["Difference"] = (
    comparison["Heavy_rain_tomorrow"]
    - comparison["No_heavy_rain_tomorrow"]
)

print(comparison.round(2).to_string())
