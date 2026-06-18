import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model():
    return joblib.load("models/champion.joblib")

model = load_model()