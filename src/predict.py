import json
import joblib
import pandas as pd
from pathlib import Path


MODEL_DIR = Path("models")

_scaler = joblib.load(MODEL_DIR / "minmax_scaler_train.joblib")
_model = joblib.load(MODEL_DIR / "champion.joblib")

_feature_cols = json.loads(
    (MODEL_DIR / "feature_cols.json").read_text()
)


def predict_yield(
    temperature: float,
    humidity: float,
    co2: float
) -> float:

    data = pd.DataFrame(
        [[temperature, humidity, co2]],
        columns=_feature_cols
    )

    scaled = _scaler.transform(data)

    prediction = _model.predict(scaled)

    return float(prediction[0])


def make_prediction(
    temperature: float,
    humidity: float,
    co2: float
):

    return predict_yield(
        temperature,
        humidity,
        co2
    )


if __name__ == "__main__":

    result = make_prediction(
        temperature=22,
        humidity=88,
        co2=920
    )

    print(f"Predicted Yield: {result:.2f} kg")