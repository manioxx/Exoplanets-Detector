import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ========================
# Load CSV and model
# ========================
df = pd.read_csv("cumulative.csv")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

features = ['koi_period','koi_duration','koi_depth','koi_prad',
            'koi_teq','koi_insol','koi_steff','koi_srad']

st.set_page_config(page_title="Exoplanet Detector", page_icon="🚀", layout="wide")
st.title("🔭 Exoplanet Detector with ML")
st.write("Select a candidate to predict if it's a confirmed exoplanet.")

# ========================
# Select candidate
# ========================
candidate_index = st.selectbox("Select candidate row", df.index)
candidate_row = df.loc[candidate_index, features]

st.subheader("Candidate Features:")
input_data = {}
for f in features:
    input_data[f] = st.number_input(f, value=float(candidate_row[f]))

# ========================
# Predict button
# ========================
if st.button("Predict"):
    feats_df = pd.DataFrame([input_data])
    feats_scaled = scaler.transform(feats_df)
    pred = rf_model.predict(feats_scaled)[0]
    conf = rf_model.predict_proba(feats_scaled)[0][1]

    st.subheader(f"Prediction: {'Exoplanet 🌍✨' if pred==1 else 'Not a planet ❌'}")
    st.progress(int(conf*100))
    st.caption(f"Confidence: {conf:.2%}")

# ========================
# Optional: show data table
# ========================
if st.checkbox("Show full dataset"):
    st.dataframe(df)
