import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# Page config
# =========================================================

st.set_page_config(page_title="Electricity Theft Detection", layout="wide")

# =========================================================
# Load model + features
# =========================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("electricity_theft_xgb_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, MODEL_FEATURES = load_artifacts()

# =========================================================
# Raw input schema (original dataset columns)
# =========================================================

RAW_FEATURES = {
    "Electricity": {
        "Electricity:Facility [kW](Hourly)": "Total facility electricity load",
        "Fans:Electricity [kW](Hourly)": "Fan electricity consumption",
        "Cooling:Electricity [kW](Hourly)": "Cooling system electricity",
        "InteriorLights:Electricity [kW](Hourly)": "Interior lighting electricity",
        "InteriorEquipment:Electricity [kW](Hourly)": "Interior equipment electricity"
    },
    "Gas": {
        "Gas:Facility [kW](Hourly)": "Total facility gas usage",
        "Heating:Gas [kW](Hourly)": "Gas used for heating",
        "InteriorEquipment:Gas [kW](Hourly)": "Gas for interior equipment",
        "Water Heater:WaterSystems:Gas [kW](Hourly)": "Water heater gas usage"
    }
}

# =========================================================
# Feature engineering (bullet-proof)
# =========================================================

def engineer_features(df):

    df = df.copy()

    # Convert everything to numeric
    df_numeric = df.apply(pd.to_numeric, errors="coerce")

    # Drop empty columns
    df_numeric = df_numeric.dropna(axis=1, how="all")

    # Fill NaNs
    df_numeric = df_numeric.fillna(df_numeric.median())

    if df_numeric.shape[1] == 0:
        raise ValueError("No numeric columns detected.")

    # Stats
    df_numeric["energy_mean"] = df_numeric.mean(axis=1)
    df_numeric["energy_std"] = df_numeric.std(axis=1)
    df_numeric["energy_max"] = df_numeric.max(axis=1)
    df_numeric["energy_min"] = df_numeric.min(axis=1)

    # Behaviour
    df_numeric["range_ratio"] = (
        df_numeric["energy_max"] - df_numeric["energy_min"]
    ) / (df_numeric["energy_mean"] + 1e-6)

    df_numeric["sudden_drop"] = (
        df_numeric.diff(axis=1).min(axis=1) < -0.3
    ).astype(int)

    df_numeric["low_usage_flag"] = (
        df_numeric["energy_mean"] < df_numeric["energy_mean"].median()
    ).astype(int)

    # Clean names
    df_numeric.columns = (
        df_numeric.columns.astype(str)
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_", regex=False)
        .str.replace(">", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    # Align to model
    for col in MODEL_FEATURES:
        if col not in df_numeric.columns:
            df_numeric[col] = 0

    return df_numeric[MODEL_FEATURES]

# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("⚙️ Controls")

mode = st.sidebar.radio("Mode", ["📁 Batch CSV", "👤 Single Customer"])

threshold = st.sidebar.slider(
    "Theft Probability Threshold",
    0.1, 0.9, 0.5, 0.05
)

st.sidebar.caption(f"Probability ≥ {threshold:.2f} → Theft")

# =========================================================
# Main
# =========================================================

st.title("⚡ Electricity Theft Detection Dashboard")

# =========================================================
# BATCH MODE
# =========================================================

if mode == "📁 Batch CSV":

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:

        raw_df = pd.read_csv(uploaded)

        st.subheader("Preview")
        st.dataframe(raw_df.head())

        X = engineer_features(raw_df)

        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)

        results = raw_df.copy()
        results["Theft_Probability"] = probs
        results["Prediction"] = np.where(preds == 1, "Theft", "Normal")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(results))
        c2.metric("Theft", (preds == 1).sum())
        c3.metric("Normal", (preds == 0).sum())

        fig, ax = plt.subplots()
        sns.countplot(x=results["Prediction"], ax=ax)
        st.pyplot(fig)

        st.dataframe(results)

        st.download_button(
            "⬇️ Download Results",
            results.to_csv(index=False).encode("utf-8"),
            "predictions.csv"
        )

    else:
        st.info("Upload a CSV to begin.")

# =========================================================
# SINGLE CUSTOMER MODE
# =========================================================

else:

    st.subheader("👤 Single Customer Input")

    autofill = None

    if "raw_df" in locals():

        st.markdown("### Auto-fill from CSV")

        idx = st.number_input(
            "Row index",
            min_value=0,
            max_value=len(raw_df) - 1,
            value=0,
            step=1
        )

        autofill = raw_df.iloc[int(idx)]

    raw_vals = {}

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("## ⚡ Electricity (kW)")

        for f, tip in RAW_FEATURES["Electricity"].items():

            default = 5.0
            if autofill is not None and f in autofill:
                default = float(autofill[f])

            raw_vals[f] = st.number_input(
                f,
                value=float(default),
                step=0.1,
                help=tip,
                key=f
            )

    with col2:

        st.markdown("## 🔥 Gas (kW)")

        for f, tip in RAW_FEATURES["Gas"].items():

            default = 2.0
            if autofill is not None and f in autofill:
                default = float(autofill[f])

            raw_vals[f] = st.number_input(
                f,
                value=float(default),
                step=0.1,
                help=tip,
                key=f
            )

    single_df = pd.DataFrame([raw_vals])

    if st.button("🔮 Predict Theft"):

        Xs = engineer_features(single_df)

        prob = model.predict_proba(Xs)[0, 1]
        pred = "Theft" if prob >= threshold else "Normal"

        st.metric("Theft Probability", f"{prob:.2%}")

        if pred == "Theft":
            st.error("🚨 POTENTIAL THEFT")
        else:
            st.success("✅ NORMAL USAGE")

        st.caption(f"Threshold: {threshold:.2f}")
