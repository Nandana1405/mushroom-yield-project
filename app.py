# app.py

import streamlit as st
import numpy as np
import pandas as pd

@st.cache_resource
def load_predict():
    from src.predict import predict_yield
    return predict_yield

predict_yield = load_predict()

st.set_page_config(
    page_title="Mushroom Yield Forecast",
    page_icon="🍄",
    layout="centered"
)

st.title("Polyhouse Yield Predictor")
st.caption("Agritech environmental forecasting from sensor data")

with st.sidebar:
    st.header("Sensor readings")

    temp = st.slider(
        "Temperature (°C)",
        10.0, 35.0, 22.0, 0.1
    )

    humid = st.slider(
        "Humidity (%)",
        50.0, 100.0, 88.0, 0.5
    )

    co2 = st.slider(
        "CO₂ (ppm)",
        400, 2000, 920, 10
    )


# Warnings
if temp < 15 or temp > 35:
    st.warning("Temperature is outside training range")

if humid < 70 or humid > 100:
    st.warning("Humidity is outside training range")

if co2 < 400 or co2 > 2000:
    st.warning("CO₂ is outside training range")

# Prediction
if st.button("Predict yield"):
    kg = predict_yield(temp, humid, co2)

    st.metric(
        label="Estimated daily yield",
        value=f"{kg:.2f} kg"
    )

# Sensitivity Chart
st.subheader("What-if: Humidity Sweep")

temp_fixed = 22.0
co2_fixed = 900

humid_range = np.linspace(70, 98, 29)

preds = [
    predict_yield(temp_fixed, h, co2_fixed)
    for h in humid_range
]

chart_df = pd.DataFrame({
    "Humidity (%)": humid_range,
    "Predicted Yield (kg)": preds
})

st.line_chart(
    chart_df,
    x="Humidity (%)",
    y="Predicted Yield (kg)"
)

# Metadata Panel
with st.expander("Model Information"):

    st.markdown("""
*Model:* Tuned Random Forest

*Training Data:* Polyhouse Sensor Data

*Features:*
- Temperature
- Humidity
- CO₂

*Output:*
- Predicted Mushroom Yield (kg)
""")