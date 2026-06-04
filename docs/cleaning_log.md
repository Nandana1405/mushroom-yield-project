## Data Cleaning Log

- Checked missing values using df.isna().sum()
- Applied validation rules:
  - humidity_pct between 50 and 100
  - temperature_c between 10 and 35
  - co2_ppm between 400 and 2000
- Removed rows with missing yield_kg
- Removed duplicate timestamps
- Saved cleaned dataset as 02_cleaned.parquet