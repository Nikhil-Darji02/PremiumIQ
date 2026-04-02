import streamlit as st
import os

st.title("Debug Mode 🚀")

st.write("Files in repo:")
st.write(os.listdir())

try:
    import pandas as pd
    st.write("Pandas loaded ✅")

    # Try loading dataset
    if "Medicalpremium.csv" in os.listdir():
        df = pd.read_csv("Medicalpremium.csv")
        st.write("CSV Loaded ✅")
    else:
        st.error("CSV NOT FOUND ❌")

    # Try loading model
    import joblib
    if "model.pkl" in os.listdir():
        model = joblib.load("model.pkl")
        st.write("Model Loaded ✅")
    else:
        st.error("MODEL NOT FOUND ❌")

except Exception as e:
    st.error(f"ERROR: {e}")
