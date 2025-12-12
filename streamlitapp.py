# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="BigMart Sales Predictor", layout="centered")

st.title("BigMart Sales Predictor")
st.write("Enter the item & outlet features and click Predict.")

# --- Inputs (match the Flask form names in your original app) ---
item_weight = st.number_input("Item weight", min_value=0.0, step=0.1, value=10.0)
item_fat_content = st.number_input("Item fat content (encoded as float)", min_value=0.0, step=0.1, value=0.0)
item_visibility = st.number_input("Item visibility", min_value=0.0, step=0.0001, format="%.6f", value=0.05)
item_type = st.number_input("Item type (encoded as float)", min_value=0.0, step=1.0, value=1.0)
item_mrp = st.number_input("Item MRP", min_value=0.0, step=0.1, value=100.0)
outlet_establishment_year = st.number_input("Outlet establishment year", min_value=1900, max_value=2100, step=1, value=1999)
outlet_size = st.number_input("Outlet size (encoded as float)", min_value=0.0, step=1.0, value=1.0)
outlet_location_type = st.number_input("Outlet location type (encoded as float)", min_value=0.0, step=1.0, value=1.0)
outlet_type = st.number_input("Outlet type (encoded as float)", min_value=0.0, step=1.0, value=1.0)

# --- Model & scaler paths (change to your actual file locations) ---
# Your Flask app used absolute Windows paths; change as needed for Streamlit environment.
SCALER_PATH = r"D:\BigMart-Sales-Prediction-using-Machine-Learning-main (1)\BigMart-Sales-Prediction-using-Machine-Learning-main\models\sc.sav"
MODEL_PATH = r"D:\BigMart-Sales-Prediction-using-Machine-Learning-main (1)\BigMart-Sales-Prediction-using-Machine-Learning-main\models\lr.sav"

# Fallback common locations (useful when running in Linux container or /mnt/data)
fallback_scaler = "/mnt/data/models/sc.sav"
fallback_model = "/mnt/data/models/lr.sav"

def _path_exists(p):
    try:
        return os.path.exists(p) and os.path.getsize(p) > 0
    except Exception:
        return False

if not _path_exists(SCALER_PATH) and _path_exists(fallback_scaler):
    SCALER_PATH = fallback_scaler

if not _path_exists(MODEL_PATH) and _path_exists(fallback_model):
    MODEL_PATH = fallback_model

@st.cache_resource
def load_scaler(path):
    return joblib.load(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

loaded = True
errors = []
if not _path_exists(SCALER_PATH):
    loaded = False
    errors.append(f"Scaler not found at: {SCALER_PATH}")
if not _path_exists(MODEL_PATH):
    loaded = False
    errors.append(f"Model not found at: {MODEL_PATH}")

if not loaded:
    st.error("Model files not found. Update the paths at top of the script.")
    for e in errors:
        st.write("- " + e)
    st.stop()

# Load resources
try:
    scaler = load_scaler(SCALER_PATH)
except Exception as e:
    st.error(f"Failed to load scaler: {e}")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Prepare features and predict
if st.button("Predict"):
    X = np.array([[
        float(item_weight),
        float(item_fat_content),
        float(item_visibility),
        float(item_type),
        float(item_mrp),
        float(outlet_establishment_year),
        float(outlet_size),
        float(outlet_location_type),
        float(outlet_type)
    ]])

    try:
        X_std = scaler.transform(X)
    except Exception as e:
        st.error(f"Error while scaling input: {e}")
        st.stop()

    try:
        y_pred = model.predict(X_std)
    except Exception as e:
        st.error(f"Error while predicting: {e}")
        st.stop()

    st.success(f"Predicted sales: {float(y_pred[0]):.4f}")
    st.write("Raw model output:", y_pred)
