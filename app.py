import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Electricity Theft Detection",
    layout="wide"
)

# ===============================
# Load artifacts
# ===============================

@st.cache_resource
def load_artifacts():
    model = joblib.load("electricity_theft_xgb_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, MODEL_FEATURES = load_artifacts()

# ===============================
# Feature Engineering
# ===============================

def engineer_features(df):

    df = df.copy()

    # Convert everything possible to numeric, force errors to NaN
    df_numeric = df.apply(pd.to_numeric, errors="coerce")

    # Drop columns that are completely NaN
    df_numeric = df_numeric.dropna(axis=1, how="all")

    # Fill remaining NaNs with column median
    df_numeric = df_numeric.fillna(df_numeric.median())

    # If user uploads empty or invalid file
    if df_numeric.shape[1] == 0:
        raise ValueError("No numeric energy columns found.")

    # Statistical features
    df_numeric["energy_mean"] = df_numeric.mean(axis=1)
    df_numeric["energy_std"] = df_numeric.std(axis=1)
    df_numeric["energy_max"] = df_numeric.max(axis=1)
    df_numeric["energy_min"] = df_numeric.min(axis=1)

    # Behavioural features
    df_numeric["range_ratio"] = (
        df_numeric["energy_max"] - df_numeric["energy_min"]
    ) / (df_numeric["energy_mean"] + 1e-6)

    df_numeric["sudden_drop"] = (
        df_numeric.diff(axis=1).min(axis=1) < -0.3
    ).astype(int)

    df_numeric["low_usage_flag"] = (
        df_numeric["energy_mean"] < df_numeric["energy_mean"].median()
    ).astype(int)

    # Clean column names
    df_numeric.columns = (
        df_numeric.columns.astype(str)
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_", regex=False)
        .str.replace(">", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    # Align with model features
    for col in MODEL_FEATURES:
        if col not in df_numeric.columns:
            df_numeric[col] = 0

    df_numeric = df_numeric[MODEL_FEATURES]

    return df_numeric



# ===============================
# Sidebar Controls
# ===============================

st.sidebar.title("⚙️ Controls")

mode = st.sidebar.radio(
    "Prediction Mode",
    ["📁 Batch CSV Upload", "👤 Single Customer"]
)

threshold = st.sidebar.slider(
    "Theft Probability Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

st.sidebar.markdown(f"""
Prediction rule:

Probability ≥ **{threshold:.2f}** → Theft  
Probability < **{threshold:.2f}** → Normal
""")

# ===============================
# Main UI
# ===============================

st.title("⚡ Electricity Theft Detection System")

# =========================================================
# BATCH MODE
# =========================================================

if mode == "📁 Batch CSV Upload":

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:

        raw_df = pd.read_csv(uploaded_file)

        st.subheader("📄 Raw Data Preview")
        st.dataframe(raw_df.head())

        X = engineer_features(raw_df)

        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)

        result_df = raw_df.copy()
        result_df["Theft_Probability"] = probs
        result_df["Theft_Prediction"] = np.where(preds == 1, "Theft", "Normal")

        # ===============================
        # KPIs
        # ===============================

        total = len(result_df)
        thefts = (preds == 1).sum()
        normals = total - thefts

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", total)
        col2.metric("Suspected Theft", thefts)
        col3.metric("Normal", normals)

        # ===============================
        # Charts
        # ===============================

        st.subheader("📊 Prediction Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(x=result_df["Theft_Prediction"], ax=ax1)
        st.pyplot(fig1)

        st.subheader("📈 Theft Probability Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(probs, bins=30, kde=True, ax=ax2)
        st.pyplot(fig2)

        # ===============================
        # Results
        # ===============================

        st.subheader("✅ Prediction Results")
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Predictions",
            csv,
            "theft_predictions.csv",
            "text/csv"
        )

    else:
        st.info("Upload a CSV file to begin.")

# =========================================================
# SINGLE CUSTOMER MODE (RESTRUCTURED UI)
# =========================================================
# =========================================================
# SINGLE CUSTOMER MODE (RESTRUCTURED UI)
# =========================================================

else:

    st.subheader("👤 Single Customer Energy Input")

    st.markdown("Enter energy readings below (you can add or remove rows).")

    if "energy_rows" not in st.session_state:
        st.session_state.energy_rows = 7

    col_add, col_remove = st.columns(2)

    if col_add.button("➕ Add Reading"):
        st.session_state.energy_rows += 1

    if col_remove.button("➖ Remove Reading"):
        if st.session_state.energy_rows > 1:
            st.session_state.energy_rows -= 1

    energy_values = []

    for i in range(st.session_state.energy_rows):
        val = st.number_input(
            label=f"Month {i+1} Energy Consumption (kWh)",
            min_value=0.0,
            step=1.0,
            key=f"energy_{i}"
        )
        energy_values.append(val)

    single_df = pd.DataFrame([energy_values])

    if st.button("🔮 Predict Theft"):

        try:

            X_single = engineer_features(single_df)

            prob = model.predict_proba(X_single)[0, 1]
            pred = "Theft" if prob >= threshold else "Normal"

            st.metric("Theft Probability", f"{prob:.2%}")

            if pred == "Theft":
                st.error("🚨 Prediction: POTENTIAL THEFT")
            else:
                st.success("✅ Prediction: NORMAL USAGE")

            st.caption(f"Decision threshold: {threshold:.2f}")

        except Exception:
            st.warning("Please enter valid numeric readings.")
