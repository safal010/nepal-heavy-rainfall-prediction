
import joblib
from pathlib import Path

model_path = Path("models/fixed_run_heavy_rain_model.pkl")
features_path = Path("models/fixed_run_model_features.pkl")

print("Checking saved files...")

print("Model file exists:", model_path.exists())
print("Feature file exists:", features_path.exists())

model = joblib.load(model_path)
features = joblib.load(features_path)

print("\nModel type:", type(model).__name__)
print("Expected features:", features)
print("Number of features:", len(features))
print("Model classes:", model.classes_)
