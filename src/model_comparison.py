import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_parquet("data/processed/features.parquet")

feature_cols = [
    "temperature_c_scaled",
    "humidity_pct_scaled",
    "co2_ppm_scaled",

]
split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

X_test = test[feature_cols]
y_test = test["yield_kg"]

# Load tuned model
model = joblib.load("models/rf_tuned.joblib")

pred = model.predict(X_test)

# Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Predicted vs Actual Yield")
plt.tight_layout()

plt.savefig(
    "reports/figures/pred_vs_actual.png",
    dpi=150
)

print("Plot saved.")