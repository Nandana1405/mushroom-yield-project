# GridSearchCV Results

## Parameter Grid

- n_estimators: [50, 100, 200]
- max_depth: [None, 8, 16]
- min_samples_leaf: [1, 3, 5]

## Best Parameters

- n_estimators: 50
- max_depth: 8
- min_samples_leaf: 3

## Performance

| Metric | Value |
|----------|---------|
| CV MAE | 0.469 |
| Test MAE | 0.44 |
| Test RMSE | 0.56 |
| Test R² | 0.371 |

## Interpretation

GridSearchCV was performed using TimeSeriesSplit with MAE scoring. The best model used 50 trees, maximum depth of 8, and a minimum leaf size of 3.

The tuned Random Forest achieved a Test MAE of 0.44 kg and Test RMSE of 0.56 kg. The cross-validation MAE was close to the test MAE, indicating stable generalization and limited overfitting.

## Saved Artifacts

- models/rf_tuned.joblib
- models/rf_best_params.json