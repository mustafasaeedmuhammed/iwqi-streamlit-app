import streamlit as st  # type: ignore
import numpy as np
import joblib
import os

st.set_page_config(page_title="IWQI / SAR / PS Predictor", layout="centered")

st.title("Irrigation Water Quality Predictor")
st.write("Enter **EC**, **pH**, and **Temperature (T)** to predict **IWQI**, **SAR**, and **PS**.")

st.info("Important: Enter EC in the SAME unit used for training (e.g., mS/cm).")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_models():
    iwqi_model = joblib.load(os.path.join(MODEL_DIR, "best_IWQI.joblib"))
    sar_model  = joblib.load(os.path.join(MODEL_DIR, "best_SAR.joblib"))
    ps_model   = joblib.load(os.path.join(MODEL_DIR, "best_PS.joblib"))
    return iwqi_model, sar_model, ps_model

iwqi_model, sar_model, ps_model = load_models()

st.subheader("Inputs")
ec = st.number_input("EC (example: mS/cm)", min_value=0.0, value=2.0, step=0.1)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.5, step=0.1)
t  = st.number_input("Temperature T (°C)", min_value=-5.0, max_value=60.0, value=25.0, step=0.5)

def iwqi_class(iwqi: float) -> str:
    if iwqi >= 85: return "NR (No restriction)"
    if iwqi >= 70: return "LR (Low restriction)"
    if iwqi >= 55: return "MR (Moderate restriction)"
    if iwqi >= 40: return "HR (High restriction)"
    return "SR (Severe restriction)"

if st.button("Predict"):
    # MUST match training order: ["EC", "PH", "T"]
    X = np.array([[ec, ph, t]], dtype=float)

    pred_iwqi = float(iwqi_model.predict(X)[0])
    pred_sar  = float(sar_model.predict(X)[0])
    pred_ps   = float(ps_model.predict(X)[0])

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("IWQI", f"{pred_iwqi:.2f}")
    c2.metric("SAR",  f"{pred_sar:.2f}")
    c3.metric("PS",   f"{pred_ps:.2f}")

    st.write("**IWQI Class:**", iwqi_class(pred_iwqi))
