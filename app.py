# =============================================================================
# INTELLIGENT PREDICTIVE MAINTENANCE DECISION SUPPORT SYSTEM
# FOR HOSPITAL INFUSION PUMPS
#
# Smart Machine Failure Prediction System
# (Machine Learning-Based Predictive Maintenance)
#
# Final-year project prototype — Department of Biomedical Engineering,
# University of Nigeria, Nsukka.
#
# This file completes the original app.py skeleton (Step 40 of the project
# notebook), which only loaded the trained model and scaler. Everything
# below — the input form, prediction logic, SHAP explainability,
# feature-importance chart, model-performance page, batch prediction, and
# safety disclaimers — is new, built on top of that same model and scaler.
#
# NOTE ON STYLING: this app deliberately uses only native Streamlit
# components (st.title, st.success, st.metric, st.warning, etc.) rather
# than raw HTML/CSS injected via st.markdown(..., unsafe_allow_html=True).
# Some hosting environments render injected HTML unreliably (or not at
# all), so native components + a .streamlit/config.toml theme are used
# instead, which work identically everywhere Streamlit runs.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# -----------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------
ASSETS_FOLDER = Path(__file__).parent / "assets"
LOGO_PATH = ASSETS_FOLDER / "app_logo.png"

st.set_page_config(
    page_title="Smart Machine Failure Prediction",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Brand palette — used only for chart colors (Plotly/Matplotlib), never
# for HTML injection, so it renders consistently everywhere.
NAVY = "#0B3D59"
TEAL = "#0E8388"
GOLD = "#D9A441"
CORAL = "#E4572E"

# -----------------------------------------------------------------------
# Load the trained model and scaler (from the actual project artifacts)
# -----------------------------------------------------------------------
MODEL_FOLDER = Path(__file__).parent / "models"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_FOLDER / "gradient_boosting_model.pkl")
    scaler = joblib.load(MODEL_FOLDER / "standard_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    ARTIFACTS_OK = True
    LOAD_ERROR = None
except Exception as e:
    ARTIFACTS_OK = False
    LOAD_ERROR = str(e)


@st.cache_resource
def load_explainer(_model):
    """TreeExplainer is fast and exact for gradient-boosted tree models like this one."""
    return shap.TreeExplainer(_model)


# The exact feature order the scaler and model were fitted on.
# (Verified directly from scaler.feature_names_in_ — do not reorder.)
FEATURE_ORDER = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_H",
    "Type_L",
    "Type_M",
]

# Real dataset-derived defaults / ranges (AI4I 2020), used for sensible
# slider bounds so a user can't submit physically nonsensical values.
FEATURE_RANGES = {
    "Air temperature [K]":     dict(min=295.0, max=305.0, default=300.0, step=0.1),
    "Process temperature [K]": dict(min=305.0, max=314.0, default=310.0, step=0.1),
    "Rotational speed [rpm]":  dict(min=1150,  max=2900,  default=1539,  step=1),
    "Torque [Nm]":             dict(min=3.0,   max=77.0,  default=40.0,  step=0.1),
    "Tool wear [min]":         dict(min=0,     max=260,   default=108,   step=1),
}

# Real, verified tuned-model performance (from GridSearchCV on the held-out
# test set — see Chapter 4 of the project report). Not placeholders.
MODEL_METRICS = {
    "Accuracy": 0.9705,
    "Precision": 0.5506,
    "Recall": 0.7206,
    "F1-Score": 0.6242,
    "ROC-AUC": 0.9629,
}
CONFUSION_MATRIX = np.array([[1892, 40], [19, 49]])
BEST_PARAMS = {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 7}


def build_feature_vector(air_temp, process_temp, rot_speed, torque, tool_wear, machine_type):
    """Builds a single-row feature vector in the exact order the scaler expects."""
    row = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_H": 1 if machine_type == "H" else 0,
        "Type_L": 1 if machine_type == "L" else 0,
        "Type_M": 1 if machine_type == "M" else 0,
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def predict(df_row):
    scaled = scaler.transform(df_row)
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][1]  # probability of class 1 (failure)
    return pred, proba


def shap_waterfall_figure(df_row, max_display=8):
    """
    Builds a SHAP waterfall plot explaining a single prediction.
    Uses TreeExplainer (exact, fast for tree ensembles) on the scaled
    feature vector, but re-labels the plot with the original,
    human-readable input values so it's interpretable by a non-technical
    reader (e.g. a biomedical engineer or lecturer).
    """
    scaled = scaler.transform(df_row)
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_ORDER)
    explainer = load_explainer(model)
    sv = explainer(scaled_df)
    sv.data = df_row.values  # show real-world values instead of scaled ones
    sv.feature_names = FEATURE_ORDER

    fig = plt.figure(figsize=(8, 4.5))
    shap.plots.waterfall(sv[0], show=False, max_display=max_display)
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------
# Header (native components only — logo image + title + caption)
# -----------------------------------------------------------------------
header_col1, header_col2 = st.columns([1, 6])
with header_col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
with header_col2:
    st.title("Smart Machine Failure Prediction System")
    st.caption("Intelligent Predictive Maintenance Decision Support — proof-of-concept "
               "prototype for hospital infusion pump maintenance")

st.divider()

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)
st.sidebar.markdown("### Smart Machine Failure Prediction")
st.sidebar.caption("Biomedical Engineering Final-Year Project · University of Nigeria, Nsukka")

if not ARTIFACTS_OK:
    st.error(
        "Could not load the trained model or scaler from the 'models' folder. Make sure "
        "gradient_boosting_model.pkl and standard_scaler.pkl are placed in a 'models' "
        "subfolder next to this app.py file, and that the installed scikit-learn version "
        "matches the one pinned in requirements.txt.\n\n"
        f"Details: {LOAD_ERROR}"
    )
    st.stop()

# -----------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------
tab_predict, tab_batch, tab_performance, tab_about = st.tabs(
    ["🔍 Single Prediction", "📄 Batch Prediction (CSV)", "📊 Model Performance", "ℹ️ About & Safety"]
)

# =========================================================================
# TAB 1 — SINGLE PREDICTION
# =========================================================================
with tab_predict:
    st.subheader("Enter Operating Parameters")
    st.caption("Values default to the AI4I 2020 dataset averages. Adjust them to reflect the "
               "equipment reading you want to evaluate.")

    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox(
            "Machine / Equipment Quality Type",
            options=["L", "M", "H"],
            index=0,
            help="L = Low, M = Medium, H = High quality variant (matches the AI4I 2020 'Type' attribute).",
        )
        air_temp = st.slider(
            "Air Temperature [K]",
            min_value=FEATURE_RANGES["Air temperature [K]"]["min"],
            max_value=FEATURE_RANGES["Air temperature [K]"]["max"],
            value=FEATURE_RANGES["Air temperature [K]"]["default"],
            step=FEATURE_RANGES["Air temperature [K]"]["step"],
        )
        process_temp = st.slider(
            "Process Temperature [K]",
            min_value=FEATURE_RANGES["Process temperature [K]"]["min"],
            max_value=FEATURE_RANGES["Process temperature [K]"]["max"],
            value=FEATURE_RANGES["Process temperature [K]"]["default"],
            step=FEATURE_RANGES["Process temperature [K]"]["step"],
        )

    with col2:
        rot_speed = st.slider(
            "Rotational Speed [rpm]",
            min_value=FEATURE_RANGES["Rotational speed [rpm]"]["min"],
            max_value=FEATURE_RANGES["Rotational speed [rpm]"]["max"],
            value=FEATURE_RANGES["Rotational speed [rpm]"]["default"],
            step=FEATURE_RANGES["Rotational speed [rpm]"]["step"],
        )
        torque = st.slider(
            "Torque [Nm]",
            min_value=FEATURE_RANGES["Torque [Nm]"]["min"],
            max_value=FEATURE_RANGES["Torque [Nm]"]["max"],
            value=FEATURE_RANGES["Torque [Nm]"]["default"],
            step=FEATURE_RANGES["Torque [Nm]"]["step"],
        )
        tool_wear = st.slider(
            "Tool Wear [min]",
            min_value=FEATURE_RANGES["Tool wear [min]"]["min"],
            max_value=FEATURE_RANGES["Tool wear [min]"]["max"],
            value=FEATURE_RANGES["Tool wear [min]"]["default"],
            step=FEATURE_RANGES["Tool wear [min]"]["step"],
        )

    predict_clicked = st.button("🔎 Predict Equipment Condition", type="primary", use_container_width=True)

    if predict_clicked:
        row = build_feature_vector(air_temp, process_temp, rot_speed, torque, tool_wear, machine_type)
        pred, proba = predict(row)

        st.markdown("### Result")
        result_col, gauge_col = st.columns([1.3, 1])

        with result_col:
            if pred == 0:
                st.success(
                    f"**✅ NORMAL OPERATION**\n\n"
                    f"The model does not currently detect a failure signature in the parameters provided.\n\n"
                    f"**Estimated failure probability: {proba*100:.2f}%**"
                )
            else:
                st.error(
                    f"**⚠️ POTENTIAL FAILURE DETECTED**\n\n"
                    f"The model flags these parameters as consistent with an elevated risk of equipment failure.\n\n"
                    f"**Estimated failure probability: {proba*100:.2f}%**"
                )

            st.warning(
                "**Reminder:** this is a decision-support estimate, not a diagnosis. A \"Normal\" "
                "result is not proof the equipment is safe, and a flagged result should trigger "
                "professional inspection rather than automatic equipment withdrawal. See the "
                "\"About & Safety\" tab for full details."
            )

        with gauge_col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba*100,
                number={'suffix': "%", 'font': {'size': 36}},
                title={'text': "Failure Probability", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': NAVY},
                    'steps': [
                        {'range': [0, 30], 'color': "#EAF4F4"},
                        {'range': [30, 60], 'color': "#FCEFE3"},
                        {'range': [60, 100], 'color': "#FBEAE6"},
                    ],
                    'threshold': {'line': {'color': CORAL, 'width': 4}, 'thickness': 0.8, 'value': 50}
                }
            ))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Why did the model predict this?")
        st.caption(
            "This SHAP (SHapley Additive exPlanations) waterfall chart shows how each input pushed "
            "the prediction toward or away from 'failure', starting from the model's average "
            "baseline output. Red bars push toward failure; blue bars push toward normal operation."
        )
        with st.spinner("Computing SHAP explanation..."):
            shap_fig = shap_waterfall_figure(row)
            st.pyplot(shap_fig, use_container_width=True)
            plt.close(shap_fig)

        with st.expander("See the exact feature vector sent to the model"):
            st.dataframe(row, use_container_width=True)

# =========================================================================
# TAB 2 — BATCH PREDICTION
# =========================================================================
with tab_batch:
    st.subheader("Batch Prediction from CSV")
    st.write(
        "Upload a CSV file with one row per equipment reading. The file must contain these exact "
        "column headers: `Type`, `Air temperature [K]`, `Process temperature [K]`, "
        "`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`."
    )

    sample = pd.DataFrame({
        "Type": ["L", "M", "H"],
        "Air temperature [K]": [300.0, 301.2, 298.9],
        "Process temperature [K]": [310.0, 311.0, 309.5],
        "Rotational speed [rpm]": [1539, 1420, 1680],
        "Torque [Nm]": [40.0, 45.2, 35.8],
        "Tool wear [min]": [108, 150, 60],
    })
    st.download_button(
        "⬇️ Download a sample CSV template",
        data=sample.to_csv(index=False),
        file_name="sample_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            required_cols = ["Type", "Air temperature [K]", "Process temperature [K]",
                              "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
            missing = [c for c in required_cols if c not in batch_df.columns]
            if missing:
                st.error(f"Your file is missing these required columns: {missing}")
            else:
                type_dummies = pd.get_dummies(batch_df["Type"], prefix="Type")
                for col in ["Type_H", "Type_L", "Type_M"]:
                    if col not in type_dummies.columns:
                        type_dummies[col] = 0
                X = pd.concat([batch_df[required_cols[1:]], type_dummies[["Type_H", "Type_L", "Type_M"]]], axis=1)
                X = X[FEATURE_ORDER]

                scaled = scaler.transform(X)
                preds = model.predict(scaled)
                probs = model.predict_proba(scaled)[:, 1]

                results = batch_df.copy()
                results["Predicted Condition"] = np.where(preds == 1, "FAILURE RISK", "NORMAL")
                results["Failure Probability (%)"] = (probs * 100).round(2)

                st.success(f"Processed {len(results)} rows.")
                st.dataframe(results, use_container_width=True)

                n_flagged = int((preds == 1).sum())
                st.info(f"⚠️ {n_flagged} of {len(results)} equipment readings were flagged as elevated failure risk.")

                st.download_button(
                    "⬇️ Download results as CSV",
                    data=results.to_csv(index=False),
                    file_name="prediction_results.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process the file: {e}")

# =========================================================================
# TAB 3 — MODEL PERFORMANCE
# =========================================================================
with tab_performance:
    st.subheader("Final Model: Tuned Gradient Boosting")
    st.write(
        "These figures come directly from evaluating the tuned model on an untouched, 2,000-row "
        "test set that was never used during training or hyperparameter tuning (see Chapter 4 of "
        "the accompanying project report)."
    )

    mcols = st.columns(5)
    metric_items = list(MODEL_METRICS.items())
    for col, (name, val) in zip(mcols, metric_items):
        with col:
            st.metric(label=name, value=f"{val*100:.2f}%")

    st.markdown("#### Hyperparameters (selected via GridSearchCV, 5-fold cross-validation)")
    st.code(
        f"n_estimators = {BEST_PARAMS['n_estimators']}\n"
        f"learning_rate = {BEST_PARAMS['learning_rate']}\n"
        f"max_depth = {BEST_PARAMS['max_depth']}",
        language="text",
    )

    st.markdown("#### Confusion Matrix (test set, n = 2,000)")
    cm_fig = px.imshow(
        CONFUSION_MATRIX,
        text_auto=True,
        color_continuous_scale=[[0, "#EAF4F4"], [1, NAVY]],
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Normal (0)", "Failure (1)"],
        y=["Normal (0)", "Failure (1)"],
    )
    cm_fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(cm_fig, use_container_width=True)

    st.markdown("#### Feature Importance")
    st.caption("Computed directly from the trained model's `feature_importances_` attribute — "
               "reflects how much each feature contributed to the model's decisions on average.")
    importances = pd.Series(model.feature_importances_, index=FEATURE_ORDER).sort_values(ascending=True)
    imp_fig = px.bar(
        importances, orientation="h",
        labels={"value": "Relative Importance", "index": ""},
        color=importances.values,
        color_continuous_scale=[[0, TEAL], [1, NAVY]],
    )
    imp_fig.update_layout(showlegend=False, coloraxis_showscale=False, height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(imp_fig, use_container_width=True)
    st.caption("Rotational speed, torque, and tool wear are the strongest drivers of the model's "
               "predictions — consistent with the correlation analysis in Chapter 4 of the report.")

# =========================================================================
# TAB 4 — ABOUT & SAFETY
# =========================================================================
with tab_about:
    st.subheader("About This Prototype")
    st.write("""
    This application is the working prototype accompanying the final-year project
    **"Intelligent Predictive Maintenance Decision Support System for Hospital Infusion Pumps: A
    Machine Learning Approach Using the Gradient Boosting Algorithm"**, submitted to the Department
    of Biomedical Engineering, Faculty of Engineering, University of Nigeria, Nsukka.
    """)

    st.markdown("#### Dataset")
    st.write("""
    The model was trained and evaluated on the **AI4I 2020 Predictive Maintenance Dataset**
    (10,000 observations, UCI Machine Learning Repository), a publicly available benchmark dataset
    simulating industrial milling-machine operating conditions. It was adopted here because
    publicly available, labelled operational datasets specifically for infusion pumps are not
    currently available.
    """)

    st.markdown("#### Model")
    st.write("""
    The deployed model is a **Gradient Boosting Classifier**, selected after comparing eight
    supervised learning algorithms and optimising the three strongest candidates (Random Forest,
    Gradient Boosting, Extra Trees) with GridSearchCV. Gradient Boosting was selected as the final
    model for its overall balance of accuracy, precision, and F1-score.
    """)

    st.markdown("#### Explainability")
    st.write("""
    Every single prediction is accompanied by a **SHAP (SHapley Additive exPlanations)** waterfall
    chart showing exactly how each input parameter pushed that specific prediction toward or away
    from "failure." This was added on the recommendation made in Chapter 5 of the project report,
    so that a biomedical engineer reviewing an alert can see *why* the model flagged a given
    reading, rather than treating it as an opaque black box.
    """)

    st.warning(
        "**⚠️ Safety and Intended Use**\n\n"
        "- This tool is a **maintenance decision-support aid**, not an autonomous clinical "
        "decision-making system, and it has **not been clinically validated** on real infusion pump data.\n"
        "- A \"Normal Operation\" prediction must **never** be treated as proof that a device is "
        "safe for clinical use.\n"
        "- A \"Failure Risk\" prediction should trigger **professional inspection**, not automatic "
        "withdrawal of equipment from service without verification.\n"
        "- Before any real clinical deployment, this model would require validation on actual "
        "infusion pump operational data, formal medical-device risk assessment, cybersecurity "
        "review, and oversight from qualified clinical and biomedical engineering personnel."
    )

    st.markdown("#### Files used by this app")
    st.code(
        "models/\n"
        "├── gradient_boosting_model.pkl   # trained, tuned Gradient Boosting classifier\n"
        "└── standard_scaler.pkl           # fitted StandardScaler (same one used in training)",
        language="text",
    )

st.divider()
st.caption("Smart Machine Failure Prediction System · Biomedical Engineering Final-Year Project · "
           "University of Nigeria, Nsukka · 2026")
