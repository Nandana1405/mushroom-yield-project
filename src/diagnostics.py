import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_parquet("data/interim/02_cleaned.parquet")
df = df.sort_values("timestamp")

feature_cols = ["temperature_c", "humidity_pct", "co2_ppm"]

# Train/Test Split
split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

# Scaling
scaler = MinMaxScaler()

X_train = scaler.fit_transform(train[feature_cols])
X_test = scaler.transform(test[feature_cols])

y_train = train["yield_kg"].values
y_test = test["yield_kg"].values

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
pred_test = model.predict(X_test)

# Residuals
residuals = y_test - pred_test

# Plot 1
plt.figure(figsize=(8,5))
plt.scatter(pred_test, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Yield")
plt.ylabel("Residual")
plt.title("Residual vs Predicted Yield")
plt.savefig("reports/figures/residual_vs_predicted.png")
plt.close()

# Plot 2
plt.figure(figsize=(8,5))
plt.scatter(X_test[:,1], residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Scaled Humidity")
plt.ylabel("Residual")
plt.title("Residual vs Humidity")
plt.savefig("reports/figures/residual_vs_humidity.png")
plt.close()

print("Diagnostic plots saved.")