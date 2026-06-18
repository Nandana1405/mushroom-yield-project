# app.py
import streamlit as st
import numpy as np
import pandas as pd
from src.predict import predict_yield

st.set_page_config(page_title="Mushroom Yield Forecast", layout="centered")

st.title("Polyhouse Yield Predictor")
st.caption("Agritech environmental forecasting from sensor data")

with st.sidebar:
    st.header("Sensor readings")
    temp = st.slider("Temperature (°C)", 10.0, 35.0, 22.0, 0.1)
    humid = st.slider("Humidity (%)", 50.0, 100.0, 88.0, 0.5)
    co2 = st.slider("CO₂ (ppm)", 400, 2000, 900, 10)

if st.button("Predict yield"):
    kg = predict_yield(temp, humid, co2)
    st.metric(label="Estimated daily yield", value=f"{kg:.2f} kg")


st.subheader("What-if: humidity sweep")
temp_fixed, co2_fixed = 22.0, 900
humid_range = np.linspace(70, 98, 29)

preds = [predict_yield(temp_fixed, h, co2_fixed) for h in humid_range]
chart_df = pd.DataFrame({"Humidity (%)": humid_range, "Predicted yield (kg)": preds})
st.line_chart(chart_df, x="Humidity (%)", y="Predicted yield (kg)")

with st.expander("Model information"):
    st.markdown("""
    - **Model:** Tuned Random Forest
    - **Test MAE:** 1.2 kg/day (example — use your value)
    - **Training data:** Polyhouse sensors Jan–Dec 2024
    """)
