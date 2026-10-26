
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load test data and trained classifier
test = pd.read_csv("data/test.csv")
model = joblib.load("models/heavy_rain_model.pkl")
features = joblib.load("models/model_features.pkl")

# Predict probabilities
X_test = test[features]
probabilities = model.predict_proba(X_test)[:, 1]

# Separate the two actual classes
actual_heavy = probabilities[
    test["heavy_rain_tomorrow"].to_numpy() == 1
]

actual_non_heavy = probabilities[
    test["heavy_rain_tomorrow"].to_numpy() == 0
]

# Plot probability distributions
plt.figure(figsize=(10, 6))

plt.hist(
    actual_non_heavy,
    bins=25,
    alpha=0.7,
    label="No heavy rainfall",
    density=True
)

plt.hist(
    actual_heavy,
    bins=15,
    alpha=0.7,
    label="Heavy rainfall",
    density=True
)

plt.xlabel("Predicted probability of heavy rainfall")
plt.ylabel("Density")
plt.title("Predicted Probabilities by Actual Rainfall Class")
plt.legend()
plt.tight_layout()

plt.savefig("data/probability_distribution.png", dpi=150)
plt.show()

print("Chart saved to data/probability_distribution.png")
