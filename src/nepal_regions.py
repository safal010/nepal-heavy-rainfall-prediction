NEPAL_REGIONS = [
    # Koshi Province
    {"location": "Biratnagar", "province": "Koshi", "belt": "Terai", "latitude": 26.4525, "longitude": 87.2718},
    {"location": "Bhadrapur", "province": "Koshi", "belt": "Terai", "latitude": 26.5440, "longitude": 88.0944},
    {"location": "Ilam", "province": "Koshi", "belt": "Hill", "latitude": 26.9094, "longitude": 87.9282},
    {"location": "Taplejung", "province": "Koshi", "belt": "Mountain", "latitude": 27.3547, "longitude": 87.6694},

    # Madhesh Province (all Terai)
    {"location": "Janakpur", "province": "Madhesh", "belt": "Terai", "latitude": 26.7288, "longitude": 85.9263},
    {"location": "Birgunj", "province": "Madhesh", "belt": "Terai", "latitude": 27.0104, "longitude": 84.8777},
    {"location": "Rajbiraj", "province": "Madhesh", "belt": "Terai", "latitude": 26.5396, "longitude": 86.7478},

    # Bagmati Province
    {"location": "Kathmandu", "province": "Bagmati", "belt": "Hill", "latitude": 27.7172, "longitude": 85.3240},
    {"location": "Hetauda", "province": "Bagmati", "belt": "Hill", "latitude": 27.4284, "longitude": 85.0322},
    {"location": "Bharatpur", "province": "Bagmati", "belt": "Terai", "latitude": 27.6833, "longitude": 84.4333},
    {"location": "Charikot", "province": "Bagmati", "belt": "Mountain", "latitude": 27.6681, "longitude": 86.0294},

    # Gandaki Province
    {"location": "Pokhara", "province": "Gandaki", "belt": "Hill", "latitude": 28.2096, "longitude": 83.9856},
    {"location": "Kawasoti", "province": "Gandaki", "belt": "Terai", "latitude": 27.6480, "longitude": 84.1220},
    {"location": "Jomsom", "province": "Gandaki", "belt": "Mountain", "latitude": 28.7806, "longitude": 83.7230},

    # Lumbini Province (Terai & Hill)
    {"location": "Butwal", "province": "Lumbini", "belt": "Terai", "latitude": 27.7006, "longitude": 83.4483},
    {"location": "Nepalgunj", "province": "Lumbini", "belt": "Terai", "latitude": 28.0500, "longitude": 81.6167},
    {"location": "Ghorahi", "province": "Lumbini", "belt": "Terai", "latitude": 28.0499, "longitude": 82.4875},
    {"location": "Tansen", "province": "Lumbini", "belt": "Hill", "latitude": 27.8673, "longitude": 83.5467},

    # Karnali Province (Hill & Mountain)
    {"location": "Birendranagar", "province": "Karnali", "belt": "Hill", "latitude": 28.6000, "longitude": 81.6167},
    {"location": "Jumla", "province": "Karnali", "belt": "Mountain", "latitude": 29.2747, "longitude": 82.1838},
    {"location": "Simikot", "province": "Karnali", "belt": "Mountain", "latitude": 29.9692, "longitude": 81.8190},

    # Sudurpashchim Province
    {"location": "Dhangadhi", "province": "Sudurpashchim", "belt": "Terai", "latitude": 28.6852, "longitude": 80.6216},
    {"location": "Dipayal", "province": "Sudurpashchim", "belt": "Hill", "latitude": 29.2610, "longitude": 80.9370},
    {"location": "Darchula", "province": "Sudurpashchim", "belt": "Mountain", "latitude": 29.8412, "longitude": 80.5446},
]

PROVINCE_ORDER = [
    "Koshi",
    "Madhesh",
    "Bagmati",
    "Gandaki",
    "Lumbini",
    "Karnali",
    "Sudurpashchim",
]

BELT_ORDER = ["Terai", "Hill", "Mountain"]


def get_all_locations(trained_cities: list[dict] | None = None) -> list[dict]:
    """Return all 24 regional locations while preserving any extra city from the model artifact."""
    trained_map = {c["location"]: c for c in (trained_cities or [])}
    merged: list[dict] = []
    seen: set[str] = set()

    for loc in NEPAL_REGIONS:
        entry = dict(loc)
        name = entry["location"]
        if name in trained_map:
            # Keep exact coordinates used during training for original cities
            entry["latitude"] = trained_map[name]["latitude"]
            entry["longitude"] = trained_map[name]["longitude"]
            entry["calibration"] = "City-specific MOS"
        else:
            entry["calibration"] = "Nepal-wide MOS"
        merged.append(entry)
        seen.add(name)

    for name, c in trained_map.items():
        if name not in seen:
            merged.append(
                {
                    "location": name,
                    "province": c.get("province", "Other"),
                    "belt": c.get("belt", "Other"),
                    "latitude": c["latitude"],
                    "longitude": c["longitude"],
                    "calibration": "City-specific MOS",
                }
            )

    return merged