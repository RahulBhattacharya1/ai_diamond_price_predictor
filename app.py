import json
import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load_models():
    reg = joblib.load("models/diamond_price_regressor.joblib")
    clf = joblib.load("models/diamond_price_range_classifier.joblib")
    with open("models/input_schema.json", "r") as f:
        schema = json.load(f)
    return reg, clf, schema

def build_input_df(schema):
    numeric_features = schema["numeric_features"]
    categorical_features = schema["categorical_features"]

    st.sidebar.header("Diamond Features")

    # Numeric inputs with reasonable ranges
    carat = st.sidebar.number_input("carat", min_value=0.0, max_value=6.0, value=0.7, step=0.01, format="%.2f")
    depth = st.sidebar.number_input("depth", min_value=40.0, max_value=80.0, value=61.5, step=0.1, format="%.1f")
    table = st.sidebar.number_input("table", min_value=40.0, max_value=85.0, value=57.0, step=0.1, format="%.1f")
    x = st.sidebar.number_input("x (mm)", min_value=0.0, max_value=12.0, value=5.7, step=0.01, format="%.2f")
    y = st.sidebar.number_input("y (mm)", min_value=0.0, max_value=12.0, value=5.7, step=0.01, format="%.2f")
    z = st.sidebar.number_input("z (mm)", min_value=0.0, max_value=12.0, value=3.5, step=0.01, format="%.2f")

    # Categorical inputs constrained to training category order
    cut = st.sidebar.selectbox("cut", schema["cut_categories"], index=schema["cut_categories"].index("Ideal") if "Ideal" in schema["cut_categories"] else 0)
    color = st.sidebar.selectbox("color", schema["color_categories"], index=schema["color_categories"].index("G") if "G" in schema["color_categories"] else 0)
    clarity = st.sidebar.selectbox("clarity", schema["clarity_categories"], index=schema["clarity_categories"].index("VS2") if "VS2" in schema["clarity_categories"] else 0)

    data = {
        "carat": [carat],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
    }

    df = pd.DataFrame(data, columns=numeric_features + categorical_features)
    return df

def main():
    st.title("Diamond Price Intelligence")
    st.write("Estimate exact price or classify a diamond into Low/Medium/High price bands.")

    reg, clf, schema = load_models()

    mode = st.radio("Choose task", ["Price Prediction", "Price Range Classification"], horizontal=True)

    input_df = build_input_df(schema)

    if st.button("Predict"):
        if mode == "Price Prediction":
            pred = reg.predict(input_df)[0]
            st.subheader("Estimated Price (USD)")
            st.metric(label="Predicted Price", value=f"${pred:,.0f}")
        else:
            band = clf.predict(input_df)[0]
            st.subheader("Estimated Price Band")
            st.metric(label="Band", value=band)
            thr = schema["price_band_thresholds"]["low_high_split"]
            st.caption(f"Bands were derived from dataset quantiles: Low ≤ {thr[0]:.0f}, Medium ≤ {thr[1]:.0f}, High > {thr[1]:.0f}")

    with st.expander("Show example input row"):
        st.dataframe(input_df)

    st.caption("Models trained with scikit-learn on the classic diamonds dataset. Preprocessing is baked into the pipelines for safe inference.")

if __name__ == "__main__":
    main()
