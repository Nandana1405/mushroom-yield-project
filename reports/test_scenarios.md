# Test Scenarios

| Scenario | Temp | Humidity | CO2 | Predicted Yield |
|-----------|------|----------|-----|----------------|
| Normal | 22 | 88 | 900 | 17.10 |
| High Humidity | 22 | 95 | 900 | 17.51 |
| Low Humidity | 22 | 70 | 900 | 16.90 |
| High Temperature | 35 | 88 | 900 | 18.22 |
| High CO2 | 22 | 88 | 2000 | 16.93 |

## Observations

- Highest predicted yield occurs at higher temperature (35°C).
- Lower humidity reduces yield.
- Higher humidity slightly improves yield compared to low humidity.
- Increased CO2 (2000 ppm) decreases predicted yield in this model.
- Temperature appears to have the strongest effect on yield.