import streamlit as st
from pycaret.classification import (
    setup, compare_models, pull, save_model,
    load_model, predict_model
)
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
import os

# Load dataset if exists
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)
else:
    df = None

# Sidebar navigation
with st.sidebar:
    st.image("plane.png", width=400)
    st.title("Passenger Satisfaction Inference App")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Inference", "Download"])

# =========================
# UPLOAD
# =========================
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload training dataset", key="upload_train")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=False)
        st.dataframe(df)

# =========================
# PROFILING
# =========================
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    if df is not None:
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)
    else:
        st.warning("Please upload a dataset first.")

# =========================
# MODELLING
# =========================
if choice == "Modelling":
    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        # Drop non-feature columns
        modelling_df = df.copy()
        if 'id' in modelling_df.columns:
            modelling_df = modelling_df.drop(columns=['id'])
        if 'Unnamed: 0' in modelling_df.columns:
            modelling_df = modelling_df.drop(columns=['Unnamed: 0'])

        chosen_target = st.selectbox("Choose the Target Column", modelling_df.columns)

        if st.button("Run Modelling"):
            try:
                setup_df = setup(
                    data=modelling_df,
                    target=chosen_target,
                    session_id=123,
                    remove_multicollinearity=True,
                    ignore_features=None
                )
                st.dataframe(pull())

                best_model = compare_models()
                st.subheader("Model Comparison")
                st.dataframe(pull())

                save_model(best_model, "best_model")
                st.success("Best model saved successfully!")

            except Exception as e:
                st.error(f"Modelling failed: {e}")
                st.warning("Check that all feature columns are numeric or categorical, "
                           "and the target column is categorical.")

# =========================
# INFERENCE
# =========================
if choice == "Inference":
    st.title("Batch Prediction (Inference)")

    if not os.path.exists("best_model.pkl"):
        st.error("No trained model found. Please run modelling first.")
    else:
        model = load_model("best_model")
        st.success("Model loaded successfully")

        test_file = st.file_uploader(
            "Upload test.csv for prediction",
            key="inference_upload"
        )

        if test_file:
            test_df = pd.read_csv(test_file)

            # Clean column names
            test_df.columns = test_df.columns.str.strip()

            # Drop index column if exists
            if 'Unnamed: 0' in test_df.columns:
                test_df = test_df.drop(columns=['Unnamed: 0'])
            if 'id' in test_df.columns:
                test_df = test_df.drop(columns=['id'])

            st.subheader("Input Data")
            st.dataframe(test_df.head())

            try:
                predictions = predict_model(model, data=test_df)
                st.subheader("Predictions")
                st.dataframe(predictions.head())

                predictions.to_csv("predictions.csv", index=False)
                st.download_button(
                    "Download Predictions",
                    predictions.to_csv(index=False),
                    "predictions.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.warning(
                    "Check that your test data columns exactly match the training dataset. "
                    "Make sure all required features are present and named correctly."
                )

# =========================
# DOWNLOAD MODEL
# =========================
if choice == "Download":
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            st.download_button('Download Model', f, file_name='best_model.pkl')
    else:
        st.warning("No model found. Please run modelling first.")

# =========================
# FOOTER
# =========================
footer = """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css" rel="stylesheet" 
integrity="sha384-gH2yIJqKdNHPEq0n4Mqa/HGKIhSkIHeL5AyhkYV8i59U5AR6csBvApHHNl/vI1Bx" crossorigin="anonymous">
<footer>
    <div style='visibility: visible;margin-top:7rem;justify-content:center;display:flex;'>
        <p style="font-size:1.1rem;">
        This is a course project lab, only for demonstration purposes.&nbsp;
        </p>
    </div>
</footer>
"""
st.markdown(footer, unsafe_allow_html=True)