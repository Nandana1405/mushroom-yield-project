from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
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


# Assume X_train, X_test, y_train, y_test from Day 8
model = LinearRegression()
model.fit(X_train, y_train)

pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
r2 = r2_score(y_test, pred_test)

print(f"Test MAE:  {mae:.2f} kg")
print(f"Test RMSE: {rmse:.2f} kg")
print(f"Test R²:   {r2:.3f}")

for name, coef in zip(["temp", "humidity", "co2"], model.coef_):
    print(f"  coef {name}: {coef:.3f}")

joblib.dump(model, "models/linear_regression.joblib")