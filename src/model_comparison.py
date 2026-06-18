import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Load models and predict with champion (example: tuned RF)
from joblib import load
import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")


feature_cols = [
    "temperature_c_scaled",
    "humidity_pct_scaled",
    "co2_ppm_scaled",
    
]

split_idx = int(len(df) * 0.8)

test = df.iloc[split_idx:]

X_test = test[feature_cols]
y_test = test["yield_kg"]

champion = load("models/rf_tuned.joblib")
pred = champion.predict(X_test)

results = pd.DataFrame({
    "model": ["Linear", "RF default", "RF tuned"],
    "test_mae": [1.8, 1.4, 1.2],  # replace with your computed values
    "test_r2": [0.55, 0.68, 0.74],
})
print(results.to_markdown(index=False))

plt.scatter(y_test, pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual yield (kg)")
plt.ylabel("Predicted yield (kg)")
plt.title("Champion Model: Predicted vs Actual")
plt.savefig("reports/figures/pred_vs_actual.png", dpi=150)