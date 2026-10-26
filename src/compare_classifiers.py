
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load the existing training and test datasets
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 2. Load the exact feature names used by the current model
features = joblib.load("models/model_features.pkl")

X_train = train[features]
y_train = train["heavy_rain_tomorrow"]

X_test = test[features]
y_test = test["heavy_rain_tomorrow"]

print("Training records:", len(train))
print("Test records:", len(test))
print("Heavy-rain cases in training:", int(y_train.sum()))
print("Heavy-rain cases in testing:", int(y_test.sum()))

# 3. Build a Logistic Regression pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),
)

# 4. Train the model
print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

# 5. Predict probabilities and classifications
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# 6. Evaluate the default 0.50 threshold
print("\n--- LOGISTIC REGRESSION RESULTS ---")
print("Threshold: 0.50")
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print(
    "Average precision:",
    round(average_precision_score(y_test, probabilities), 4),
)
print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))
print("F1-score:", round(f1_score(y_test, predictions, zero_division=0), 4))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

# 7. Save this model separately; do not overwrite the Random Forest model
joblib.dump(model, "models/logistic_regression_heavy_rain.pkl")
print("\nModel saved to models/logistic_regression_heavy_rain.pkl")
