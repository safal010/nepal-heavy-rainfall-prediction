
import pandas as pd
import joblib

# Load the trained heavy-rainfall classification model
model = joblib.load("models/heavy_rain_model.pkl")

# Load the feature names used during training
features = joblib.load("models/model_features.pkl")

# Get feature importance scores
importance = model.feature_importances_

# Combine names and scores into a table
importance_df = pd.DataFrame({
    "feature": features,
    "importance": importance
})

# Sort from most important to least important
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\n--- HEAVY RAINFALL FEATURE IMPORTANCE ---")
print(importance_df.to_string(index=False))

# Save the results
importance_df.to_csv(
    "data/feature_importance.csv",
    index=False
)

print("\nResults saved to data/feature_importance.csv")
