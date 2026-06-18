import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model():
    return joblib.load("models/champion.joblib")

model = load_model()

def predict_yield(temp, humid, co2):
    X = pd.DataFrame({
        "temperature_c_scaled": [temp],
        "humidity_pct_scaled": [humid],
        "co2_ppm_scaled": [co2]
    })

    prediction = model.predict(X)
    return prediction[0]