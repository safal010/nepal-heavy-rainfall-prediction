
import pandas as pd
import joblib
import matplotlib.pyplot as plt

test = pd.read_csv("data/test.csv")
model = joblib.load("models/heavy_rain_model.pkl")
features = joblib.load("models/model_features.pkl")

X_test = test[features]
probabilities = model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    "Actual class": test["heavy_rain_tomorrow"].map({
        0: "No heavy rainfall",
        1: "Heavy rainfall"
    }),
    "Probability": probabilities
})

groups = [
    results.loc[
        results["Actual class"] == "No heavy rainfall",
        "Probability"
    ],
    results.loc[
        results["Actual class"] == "Heavy rainfall",
        "Probability"
    ]
]

plt.figure(figsize=(8, 6))
plt.boxplot(
    groups,
    tick_labels=["No heavy rainfall", "Heavy rainfall"]
)

plt.ylabel("Predicted probability")
plt.title("Predicted Probability by Actual Class")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("data/probability_boxplot.png", dpi=150)
plt.show()

print("Chart saved to data/probability_boxplot.png")
