Mushroom Yield Prediction

This project aims to predict mushroom yield using environmental factors such as temperature, humidity, and CO₂ levels. It is an agritech machine learning project that will use data analysis and predictive modeling to help improve mushroom production.

Environment Setup

1. Installed Python

2. Created virtual environment (venv)

3. Installed pandas, numpy, matplotlib, scikit-learn and jupyter

4. Ran test.py successfully

Project Structure

data/

src/

models/

notebooks/
\# Environment Setup



1\. Installed Python

2\. Created virtual environment (venv)

3\. Installed pandas, numpy, matplotlib, scikit-learn and jupyter

4\. Ran test.py successfully
## Project Structure
data/
src/
models/
notebooks/
## Column Definitions

| Column | Definition | Unit |
|----------|------------|------|
| timestamp | Date and time when sensor data was recorded | Date/Time |
| temperature_c | Temperature inside the polyhouse | °C |
| humidity_pct | Relative humidity inside the polyhouse | % |
| co2_ppm | Carbon dioxide concentration | ppm |
| yield_kg | Mushroom yield harvested | kg |
### Output Files

* data/interim/02_cleaned.parquet
* docs/cleaning_log.md

Feature Definitions

temperature_c:
Temperature sensor reading in °C

humidity_pct:
Humidity sensor reading in %

co2_ppm:
CO₂ concentration in ppm

temp_humid_interaction:
temperature_c × humidity_pct / 100

Train/Test Split

Dataset sorted by timestamp.
80% used for training.
20% used for testing.

Training data was used to fit MinMaxScaler.
Testing data was transformed using the same scaler to avoid data leakage.

### Split Summary

Train Rows: 292

Test Rows: 73

Train Dates:
2024-01-01 to 2024-10-19

Test Dates:
2024-10-20 to 2024-12-31


## Linear Regression Interpretation

Temperature coefficient:
Positive value means higher temperature increases yield.

Humidity coefficient:
Positive value means higher humidity increases yield.

CO2 coefficient:
Negative value means higher CO2 decreases yield.

## Residual Analysis

Diagnostic plots generated:

- residual_vs_predicted.png
- residual_vs_humidity.png

Observations:

- Residuals are centered around zero.
- No strong systematic pattern observed.
- Linear Regression is acceptable as a baseline model.

## Random Forest Results

Random Forest model was trained and compared with the baseline Linear Regression model.
 
 # comparison
 
RF CV MAE: 0.47404833333333257 +/- 0.05711333157026999
Linear CV MAE: 0.4405590208164414 +/- 0.03375818091090288

Outputs:
- models/random_forest.joblib
- reports/random_forest_report.md
- reports/cv_results.md
- reports/figures/rf_importance.png

## Model Comparison
Model	Test MAE	Test R²
Linear Regression	1.8	0.55
Random Forest (Default)	1.4	0.68
Random Forest (Tuned)	1.2	0.74

The Tuned Random Forest achieved the lowest MAE and highest R² score, indicating superior predictive performance.

Outputs:
-models/rf_tuned.joblib
-models/champion.joblib
-models/rf_best_params.json
-reports/grid_search_results.md
-reports/model_comparison.md
-src/grid_search.py
-src/model_comparison.py

## Run Inference

Predict mushroom yield:

```bash
python src/predict.py