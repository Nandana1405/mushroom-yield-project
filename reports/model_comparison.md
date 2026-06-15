# Model Comparison

| Model | Test MAE | Test RMSE | Test R² |
|---------|---------|---------|---------|
| Linear Regression | 0.46 | 0.58 | 0.34 |
| Random Forest (Default) | 0.45 | 0.57 | 0.33 |
| Random Forest (Tuned) | 0.44 | 0.56 | 0.37 |

## Champion Model

Random Forest (Tuned) was selected as the champion model because it achieved the lowest MAE and RMSE while obtaining the highest R² score on the untouched test set.

The model generalized well during TimeSeriesSplit cross-validation and showed limited signs of overfitting.