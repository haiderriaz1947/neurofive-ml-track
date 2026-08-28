"""
Customer Churn Predictor — Streamlit Web App
---------------------------------------------
Loads a saved scikit-learn Pipeline (churn_ensemble_pipeline.pkl) trained on the
Telco Customer Churn dataset and serves live predictions through a simple UI.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy:
    Push this file + requirements.txt + churn_ensemble_pipeline.pkl to a GitHub repo,
    then deploy on https://share.streamlit.io (Streamlit Community Cloud).
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📞",
    layout="centered",
)

MODEL_PATH = "churn_ensemble_pipeline.pkl"  # <-- change if your filename differs


# --------------------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# --------------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    try:
        model = joblib.load(path)
        return model, None
    except FileNotFoundError:
        return None, f"Model file not found: '{path}'. Make sure it sits next to app.py."
    except Exception as e:
        return None, f"Could not load model: {e}"


model, load_error = load_model(MODEL_PATH)

st.title("📞 Customer Churn Predictor")
st.write(
    "Enter a customer's details below and click **Predict** to estimate the "
    "probability that they will churn."
)

if load_error:
    st.error(load_error)
    st.info(
        "Place your saved `.pkl` file (e.g. `churn_ensemble_pipeline.pkl`) in the "
        "same folder as `app.py`, then rerun the app."
    )
    st.stop()

# Sanity check: warn if the input form won't match what the pipeline expects.
expected_cols = getattr(model, "feature_names_in_", None)
if expected_cols is not None:
    with st.expander("🔍 Columns this model expects (debug info)"):
        st.write(list(expected_cols))

# --------------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------------
with st.form("churn_form"):
    st.subheader("Customer Profile")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=0.5
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=1.0
        )

    submitted = st.form_submit_button("🔮 Predict")

# --------------------------------------------------------------------------------
# Build input row + predict
# --------------------------------------------------------------------------------
if submitted:
    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # --- Engineered features (added by churn_pipeline_feature_engineering.ipynb) ---
    # NOTE: these formulas are best-guess reconstructions. Verify against your
    # notebook's actual feature-engineering code and adjust if they differ.
    service_cols_yes = [
        phone_service, multiple_lines, online_security, online_backup,
        device_protection, tech_support, streaming_tv, streaming_movies,
    ]
    num_services = sum(1 for v in service_cols_yes if v == "Yes")
    if internet_service != "No":
        num_services += 1
    input_dict["NumServices"] = num_services

    input_dict["AvgMonthlyCharge"] = (
        total_charges / tenure if tenure > 0 else monthly_charges
    )

    input_dict["IsNewCustomer"] = 1 if tenure <= 6 else 0
    # ---------------------------------------------------------------------------

    input_df = pd.DataFrame([input_dict])

    # If the pipeline remembers its training columns, align/validate against them
    # so a naming mismatch fails loudly instead of silently mispredicting.
    if expected_cols is not None:
        missing = [c for c in expected_cols if c not in input_df.columns]
        extra = [c for c in input_df.columns if c not in expected_cols]
        if missing:
            st.error(
                "Your saved pipeline expects columns this form doesn't provide: "
                f"{missing}. Open app.py and edit the `input_dict` / form fields "
                "to match your notebook's training columns exactly."
            )
            st.stop()
        if extra:
            input_df = input_df[list(expected_cols)]  # drop unused extras, reorder

    try:
        prediction = model.predict(input_df)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]

        st.divider()
        st.subheader("Result")

        churn_label = "Yes" if str(prediction) in ("1", "Yes", "True") else "No"

        if churn_label == "Yes":
            st.error("⚠️ This customer is **likely to churn**.")
        else:
            st.success("✅ This customer is **likely to stay**.")

        if proba is not None:
            classes = list(model.classes_)
            churn_idx = None
            for candidate in (1, "Yes", "1", True):
                if candidate in classes:
                    churn_idx = classes.index(candidate)
                    break
            if churn_idx is None:
                churn_idx = int(np.argmax(proba))  # fallback

            churn_prob = proba[churn_idx]
            st.metric("Churn Probability", f"{churn_prob:.1%}")
            st.progress(min(max(churn_prob, 0.0), 1.0))

        with st.expander("See the input sent to the model"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "This usually means the form's columns/categories don't exactly match "
            "what the pipeline was trained on. Check `feature_names_in_` above."
        )

st.divider()
st.caption(
    "Model: churn_ensemble_pipeline.pkl · Built with Streamlit · "
    "Data schema: Telco Customer Churn dataset"
)
