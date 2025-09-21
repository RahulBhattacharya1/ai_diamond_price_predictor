import json
import pandas as pd
import streamlit as st
import skops.io as sio

st.set_page_config(page_title="Diamond Price Intelligence")

# --- Load models into session state ---
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

if st.button("Load models") or st.session_state.models_loaded:
    if not st.session_state.models_loaded:
        reg = sio.load("models/diamond_price_regressor.skops")
        clf = sio.load("models/diamond_price_range_classifier.skops")
        with open("models/input_schema.json", "r") as f:
            schema = json.load(f)
        st.session_state.reg = reg
        st.session_state.clf = clf
        st.session_state.schema = schema
        st.session_state.models_loaded = True
    st.success("✅ Models are loaded and ready!")

# --- If models not loaded, stop here ---
if not st.session_state.models_loaded:
    st.warning("Click **Load models** to start.")
    st.stop()

# --- Build input form ---
schema = st.session_state.schema
numeric_features = schema["numeric_features"]
categorical_features = schema["categorical_features"]

st.sidebar.header("Diamond Features")
carat = st.sidebar.number_input("carat", 0.0, 6.0, 0.7, 0.01)
depth = st.sidebar.number_input("depth", 40.0, 80.0, 61.5, 0.1)
table = st.sidebar.number_input("table", 40.0, 85.0, 57.0, 0.1)
x = st.sidebar.number_input("x (mm)", 0.0, 12.0, 5.7, 0.01)
y = st.sidebar.number_input("y (mm)", 0.0, 12.0, 5.7, 0.01)
z = st.sidebar.number_input("z (mm)", 0.0, 12.0, 3.5, 0.01)

cut = st.sidebar.selectbox("cut", schema["cut_categories"], index=4)
color = st.sidebar.selectbox("color", schema["color_categories"], index=3)
clarity = st.sidebar.selectbox("clarity", schema["clarity_categories"], index=3)

input_df = pd.DataFrame([{
    "carat": carat, "depth": depth, "table": table,
    "x": x, "y": y, "z": z,
    "cut": cut, "color": color, "clarity": clarity
}], columns=numeric_features + categorical_features)

# --- Prediction mode ---
mode = st.radio("Choose task", ["Price Prediction", "Price Range Classification"], horizontal=True)

if st.button("Predict"):
    if mode == "Price Prediction":
        pred = st.session_state.reg.predict(input_df)[0]
        st.subheader("Estimated Price (USD)")
        st.metric("Predicted Price", f"${pred:,.0f}")
    else:
        band = st.session_state.clf.predict(input_df)[0]
        st.subheader("Estimated Price Band")
        st.metric("Band", band)
