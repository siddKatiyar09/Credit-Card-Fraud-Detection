from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from tensorflow import keras
except ImportError:
    keras = None


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
FEATURE_COLUMNS = ["Time", "Amount", *[f"V{index}" for index in range(1, 29)], "LogAmount", "Hour", "DayIndex"]


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    frame["LogAmount"] = np.log1p(frame["Amount"].clip(lower=0))
    frame["Hour"] = ((frame["Time"] // 3600) % 24).astype(int)
    frame["DayIndex"] = (frame["Time"] // 86400).astype(int)
    return frame


@st.cache_resource
def load_baseline_bundle() -> dict:
    with (MODELS_DIR / "baseline_models.pkl").open("rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_deep_learning_assets() -> tuple[dict, object | None]:
    preprocessor_path = MODELS_DIR / "deep_learning_preprocessor.pkl"
    model_path = MODELS_DIR / "fraud_neural_network.keras"

    if not preprocessor_path.exists() or not model_path.exists() or keras is None:
        return {}, None

    with preprocessor_path.open("rb") as file:
        preprocessor_bundle = pickle.load(file)
    model = keras.models.load_model(model_path)
    return preprocessor_bundle, model


def predict_with_baseline(dataframe: pd.DataFrame, bundle: dict, selected_model: str) -> tuple[np.ndarray, float]:
    if selected_model == "Random Forest":
        model = bundle["random_forest_model"]
    else:
        model = bundle["logistic_pipeline"]
        selected_model = "Logistic Regression + SMOTE"
    threshold = bundle["thresholds"][selected_model]
    probabilities = model.predict_proba(dataframe[FEATURE_COLUMNS])[:, 1]
    return probabilities, threshold


def predict_with_neural_network(dataframe: pd.DataFrame, bundle: dict, model: object) -> tuple[np.ndarray, float]:
    transformed = bundle["preprocessor"].transform(dataframe[FEATURE_COLUMNS])
    probabilities = model.predict(transformed, verbose=0).ravel()
    return probabilities, bundle["threshold"]


st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("Credit Card Fraud Detection App")
st.write("Upload transactions, score fraud probability, and download flagged results.")

uploaded_file = st.file_uploader("Upload `creditcard.csv` style data", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

inference_df = pd.read_csv(uploaded_file)
required_columns = {"Time", "Amount", *[f"V{index}" for index in range(1, 29)]}
missing_columns = required_columns.difference(inference_df.columns)
if missing_columns:
    st.error(f"Missing required columns: {', '.join(sorted(missing_columns))}")
    st.stop()

inference_df = engineer_features(inference_df)
baseline_bundle = load_baseline_bundle()
deep_learning_bundle, deep_learning_model = load_deep_learning_assets()

available_models = ["Logistic Regression + SMOTE", "Random Forest"]
if deep_learning_model is not None:
    available_models.append("Deep Neural Network")

selected_model = st.selectbox("Choose a model", available_models)

if st.button("Predict Fraud", type="primary"):
    if selected_model == "Deep Neural Network":
        probabilities, threshold = predict_with_neural_network(inference_df, deep_learning_bundle, deep_learning_model)
    else:
        probabilities, threshold = predict_with_baseline(inference_df, baseline_bundle, selected_model)

    results = inference_df.copy()
    results["fraud_probability"] = probabilities
    results["fraud_prediction"] = (results["fraud_probability"] >= threshold).astype(int)

    st.subheader("Prediction Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows Scored", f"{len(results):,}")
    col2.metric("Predicted Fraud", f"{int(results['fraud_prediction'].sum()):,}")
    col3.metric("Average Fraud Probability", f"{results['fraud_probability'].mean():.4f}")

    st.dataframe(results.head(50), use_container_width=True)
    st.download_button(
        label="Download Results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )
