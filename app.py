import streamlit as st  # type: ignore
import numpy as np
import joblib
import os
from PIL import Image

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="IWQI / SAR / PS Predictor", layout="centered")

# -----------------------------
# Header logos (from GitHub /assets)
# Your current filenames end with .png.jpg, so we must use them exactly.
# -----------------------------
@st.cache_resource
def load_header_images():
    logo = Image.open(os.path.join(os.path.dirname(__file__), "assets", "logo_university.png.jpg"))
    banner = Image.open(os.path.join(os.path.dirname(__file__), "assets", "header_banner.png.jpg"))
    return logo, banner

try:
    logo_img, banner_img = load_header_images()
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(logo_img, width=110)
    with col2:
        st.image(banner_img, use_container_width=True)
    st.markdown("---")
except Exception as e:
    # If images are missing or paths are wrong, app still runs.
    st.warning(f"Header images not loaded: {e}")

# -----------------------------
# App title & description
# -----------------------------
st.title("Irrigation Water Quality Predictor for Muravera Coastal Aquifer")
st.write("Enter **EC**, **pH**, and **Temperature (T)** to predict **IWQI**, **SAR**, and **PS**.")
st.info("Important: Enter EC in the SAME unit used for training (e.g., mS/cm).")

# -----------------------------
# Models
# -----------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_models():
    iwqi_model = joblib.load(os.path.join(MODEL_DIR, "best_IWQI.joblib"))
    sar_model  = joblib.load(os.path.join(MODEL_DIR, "best_SAR.joblib"))
    ps_model   = joblib.load(os.path.join(MODEL_DIR, "best_PS.joblib"))
    return iwqi_model, sar_model, ps_model

iwqi_model, sar_model, ps_model = load_models()

# -----------------------------
# Inputs
# -----------------------------
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

# -----------------------------
# Predict
# -----------------------------
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

