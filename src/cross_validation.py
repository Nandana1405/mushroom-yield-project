import pandas as pd

df = pd.read_parquet("data/interim/02_cleaned.parquet")
df = df.sort_values("timestamp")
feature_cols = [
    "temperature_c",
    "humidity_pct",
    "co2_ppm"
]

X = df[feature_cols]
y = df["yield_kg"]

split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

tscv = TimeSeriesSplit(n_splits=5)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
lin = LinearRegression()

rf_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")
lin_scores = cross_val_score(lin, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")

print("RF CV MAE:", (-rf_scores).mean(), "+/-", (-rf_scores).std())
print("Linear CV MAE:", (-lin_scores).mean(), "+/-", (-lin_scores).std())