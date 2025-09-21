import json
import os
import traceback
import joblib
import pandas as pd
import streamlit as st

def env_report():
    import sklearn, numpy, pandas
    return {
        "sklearn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "joblib": joblib.__version__,
    }

def safe_load_models():
    try:
        reg = joblib.load("models/diamond_price_regressor.joblib")
        clf = joblib.load("models/diamond_price_range_classifier.joblib")
        with open("models/input_schema.json", "r") as f:
            schema = json.load(f)
        return reg, clf, schema, None
    except Exception as e:
        return None, None, None, "".join(traceback.format_exception_only(type(e), e)).strip()

def build_input_df(schema):
    numeric_features = schema["numeric_features"]
    categorical_features = schema["categorical_features"]

    st.sidebar.header("Diamond Features")

    carat = st.sidebar.number_input("carat", 0.0, 6.0, 0.7, 0.01, format="%.2f")
    depth = st.sidebar.number_input("depth", 40.0, 80.0, 61.5, 0.1, format="%.1f")
    table = st.sidebar.number_input("table", 40.0, 85.0, 57.0, 0.1, format="%.1f")
    x = st.sidebar.number_input("x (mm)", 0.0, 12.0, 5.7, 0.01, format="%.2f")
    y = st.sidebar.number_input("y (mm)", 0.0, 12.0, 5.7, 0.01, format="%.2f")
    z = st.sidebar.number_input("z (mm)", 0.0, 12.0, 3.5, 0.01, format="%.2f")

    cut = st.sidebar.selectbox("cut", ["Fair","Good","Very Good","Premium","Ideal"], index=4)
    color = st.sidebar.selectbox("color", ["J","I","H","G","F","E","D"], index=3)
    clarity = st.sidebar.selectbox("clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"], index=3)

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
    return pd.DataFrame(data, columns=numeric_features + categorical_features)

def main():
    st.title("Diamond Price Intelligence")

    with st.expander("Environment versions (diagnostics)"):
        st.json(env_report())

    if not os.path.exists("models"):
        st.error("The 'models' folder is missing in the repo.")
        return

    mode = st.radio("Choose task", ["Price Prediction", "Price Range Classification"], horizontal=True)

    load_now = st.button("Load models")
    if not load_now:
        st.info("Click 'Load models' to initialize the predictors.")
        return

    reg, clf, schema, err = safe_load_models()
    if err:
        st.error("Model loading failed.\n\n" + err)
        st.warning(
            "If you trained in Colab with scikit-learn 1.4.2 but Streamlit uses a different version, "
            "pin exact versions in requirements.txt as shown in the guide, then reboot the app."
        )
        return

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
            st.caption(f"Bands from training quantiles: Low ≤ {thr[0]:.0f}, Medium ≤ {thr[1]:.0f}, High > {thr[1]:.0f}")

    with st.expander("Show example input row"):
        st.dataframe(input_df)

if __name__ == "__main__":
    main()
