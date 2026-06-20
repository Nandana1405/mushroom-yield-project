# Monitoring Plan

## Prediction Logging

The application should log the following information for each prediction:

* Timestamp
* Temperature (°C)
* Humidity (%)
* CO2 (ppm)
* Predicted Yield (kg)

### Example Log Entry

| Timestamp        | Temperature | Humidity | CO2 | Predicted Yield |
| ---------------- | ----------- | -------- | --- | --------------- |
| 2026-06-20 23:00 | 22          | 88       | 900 | 17.10           |

## Retraining Triggers

The model should be retrained when:

1. More than 1000 new records are collected.
2. Prediction accuracy drops significantly.
3. Sensor data distribution changes noticeably.
4. Monthly performance review indicates degradation.

## Monitoring Frequency

* Daily review of prediction logs.
* Weekly review of data quality.
* Monthly model performance evaluation.
