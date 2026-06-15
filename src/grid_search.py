import pandas as pd
import numpy as np
import json
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")

y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

# Time Series CV
tscv = TimeSeriesSplit(n_splits=5)

# Parameter Grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 8, 16],
    "min_samples_leaf": [1, 3, 5]
}

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

search = GridSearchCV(
    rf,
    param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train)

print("Best Parameters:", search.best_params_)
print("Best CV MAE:", -search.best_score_)

best_model = search.best_estimator_

pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"Test MAE: {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R2: {r2:.3f}")

joblib.dump(best_model, "models/rf_tuned.joblib")

with open("models/rf_best_params.json", "w") as f:
    json.dump(search.best_params_, f, indent=2)