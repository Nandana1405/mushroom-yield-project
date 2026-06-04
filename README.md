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
## Data Cleaning Log

- Checked missing values using df.isna().sum()
- Applied validation rules:
  - humidity_pct between 50 and 100
  - temperature_c between 10 and 35
  - co2_ppm between 400 and 2000
- Removed rows with missing yield_kg
- Removed duplicate timestamps
- Saved cleaned dataset as 02_cleaned.parquet