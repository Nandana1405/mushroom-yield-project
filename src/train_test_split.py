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