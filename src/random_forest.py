from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
import joblib

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_parquet("data/interim/02_cleaned.parquet").sort_values("timestamp")

feature_cols = ["temperature_c", "humidity_pct", "co2_ppm"]

split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

scaler = MinMaxScaler()

X_train = scaler.fit_transform(train[feature_cols])
X_test = scaler.transform(test[feature_cols])

y_train = train["yield_kg"].values
y_test = test["yield_kg"].values

joblib.dump(scaler, "models/minmax_scaler_train.joblib")

print(f"Train rows: {len(train)}")
print(f"Test rows: {len(test)}")

print(f"Train dates: {train['timestamp'].min()} to {train['timestamp'].max()}")
print(f"Test dates: {test['timestamp'].min()} to {test['timestamp'].max()}")

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)
print(f"RF Test MAE: {mean_absolute_error(y_test, pred):.2f} kg")
print(f"RF Test R²:  {r2_score(y_test, pred):.3f}")
print(f"RF Test RMSE: {np.sqrt(mean_squared_error(y_test,pred)):.2f}kg")

importances = rf.feature_importances_
labels = ["temperature", "humidity", "co2"]
plt.barh(labels, importances)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("reports/figures/rf_importance.png", dpi=150)

joblib.dump(rf, "models/random_forest.joblib")