import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model():
    model = joblib.load("models/random_forest.joblib")
    scaler = joblib.load("models/minmax_scaler_train.joblib")
    return model, scaler

model, scaler = load_model()

def predict_yield(temp, humid, co2):

    X = pd.DataFrame({
        "temperature_c": [temp],
        "humidity_pct": [humid],
        "co2_ppm": [co2]
    })

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)

    return float(prediction[0])