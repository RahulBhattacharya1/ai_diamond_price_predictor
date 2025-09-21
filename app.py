import json
import os
import traceback
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diamond Price Intelligence")

# Try using skops for loading; if not available, we still can train on the fly
USE_SKOPS = True
try:
    import skops.io as sio
except Exception:
    USE_SKOPS = False

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# Fixed schema for consistent preprocessing
NUMERIC = ["carat", "depth", "table", "x", "y", "z"]
CATEG = ["cut", "color", "clarity"]
CUT_CATS = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_CATS = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_CATS = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

def build_preprocessor():
    num_tr = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tr = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 categories=[CUT_CATS, COLOR_CATS, CLARITY_CATS]))
    ])
    return ColumnTransformer([("num", num_tr, NUMERIC),
                              ("cat", cat_tr, CATEG)])

@st.cache_resource(show_spinner=False)
def load_or_train_models():
    # 1) Try loading .skops artifacts
    schema = {
        "numeric_features": NUMERIC,
        "categorical_features": CATEG,
        "cut_categories": CUT_CATS,
        "color_categories": COLOR_CATS,
        "clarity_categories": CLARITY_CATS,
    }
    reg, clf = None, None
    errors = []

    if USE_SKOPS:
        try:
            reg = sio.load("models/diamond_price_regressor.skops", trusted=True)
            clf = sio.load("models/diamond_price_range_classifier.skops", trusted=True)
            # If there is a JSON, read thresholds from it; else compute after training fallback
            if os.path.exists("models/input_schema.json"):
                with open("models/input_schema.json", "r") as f:
                    j = json.load(f)
                    schema.update(j)
            return reg, clf, schema, None
        except Exception as e:
            errors.append("SKOPS load failed: " + "".join(traceback.format_exception_only(type(e), e)).strip())

    # 2) Fallback: train quickly from diamonds.csv and cache the models
    try:
        df = pd.read_csv("diamonds.csv")
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        pre = build_preprocessor()

        # Regressor
        X = df[NUMERIC + CATEG]
        y = df["price"].astype(float)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = Pipeline([("preprocess", pre),
                        ("model", HistGradientBoostingRegressor(max_depth=6, max_iter=200,
                                                                learning_rate=0.08, random_state=42))])
        reg.fit(Xtr, ytr)
        _ = (mean_absolute_error(yte, reg.predict(Xte)), r2_score(yte, reg.predict(Xte)))  # optional

        # Classifier
        q1, q2 = df["price"].quantile([0.33, 0.66])
        def band(p): return "Low" if p <= q1 else ("Medium" if p <= q2 else "High")
        df["price_band"] = df["price"].apply(band)

        Xc = df[NUMERIC + CATEG]
        yc = df["price_band"]
        Xctr, Xcte, yctr, ycte = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
        clf = Pipeline([("preprocess", pre),
                        ("model", HistGradientBoostingClassifier(max_depth=6, max_iter=250,
                                                                 learning_rate=0.08, random_state=42))])
        clf.fit(Xctr, yctr)
        _ = accuracy_score(ycte, clf.predict(Xcte))  # optional

        schema["price_band_thresholds"] = {"low_high_split": [float(q1), float(q2)]}
        return reg, clf, schema, None
    except Exception as e:
        errors.append("Fallback training failed: " + "".join(traceback.format_exception_only(type(e), e)).strip())
        return None, None, None, "\n".join(errors)

# Session state
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

st.title("Diamond Price Intelligence")

# Load once
if not st.session_state.models_loaded:
    with st.spinner("Initializing models..."):
        reg, clf, schema, err = load_or_train_models()
    if err:
        st.error("Initialization failed:\n\n" + err)
        st.stop()
    st.session_state.reg = reg
    st.session_state.clf = clf
    st.session_state.schema = schema
    st.session_state.models_loaded = True
    st.success("Models are ready.")

# UI
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
        if "price_band_thresholds" in schema:
            thr = schema["price_band_thresholds"]["low_high_split"]
            st.caption(f"Bands: Low ≤ {thr[0]:.0f}, Medium ≤ {thr[1]:.0f}, High > {thr[1]:.0f}")
