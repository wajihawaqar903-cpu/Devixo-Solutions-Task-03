
import streamlit as st
import pandas as pd
import joblib

# Load the saved model files
model = joblib.load("best_model.joblib")
scaler = joblib.load("scaler.joblib")
column_names = joblib.load("column_names.joblib")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")

st.title("Telco Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of churn.")

# Customer details form
with st.form("customer_form"):

    st.subheader("Customer Information")

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=72, value=12
    )

    st.subheader("Services")

    phone = st.selectbox("Phone Service", ["Yes", "No"])

    multiple_lines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup", ["No", "Yes", "No internet service"]
    )

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

    st.subheader("Billing Information")

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0
    )

    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=840.0
    )

    predict_button = st.form_submit_button("Predict Churn")


if predict_button:

    # Build the input row
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Apply the same feature engineering
    customer["TotalCharges"] = pd.to_numeric(
        customer["TotalCharges"], errors="coerce"
    )
    customer["TotalCharges"] = customer["TotalCharges"].fillna(
        customer["TotalCharges"].median()
    )

    customer["TenureGroup"] = pd.cut(
        customer["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["0-12 months", "13-24 months",
                "25-48 months", "49+ months"]
    )

    customer["ChargesPerTenureMonth"] = (
        customer["MonthlyCharges"] * customer["tenure"]
    )

    # Preserve the original dataset's category levels.
    # This keeps one-hot encoding consistent for a single customer.
    category_levels = {
        "gender": ["Female", "Male"],
        "Partner": ["No", "Yes"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "No internet service", "Yes"],
        "OnlineBackup": ["No", "No internet service", "Yes"],
        "DeviceProtection": ["No", "No internet service", "Yes"],
        "TechSupport": ["No", "No internet service", "Yes"],
        "StreamingTV": ["No", "No internet service", "Yes"],
        "StreamingMovies": ["No", "No internet service", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["No", "Yes"],
        "PaymentMethod": [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ],
        "TenureGroup": [
            "0-12 months", "13-24 months",
            "25-48 months", "49+ months"
        ]
    }

    for col, levels in category_levels.items():
        customer[col] = pd.Categorical(
            customer[col], categories=levels
        )

    # One-hot encoding
    customer_encoded = pd.get_dummies(
        customer, drop_first=True, dtype=int
    )

    # Keep exactly the model's features and their order
    customer_encoded = customer_encoded.reindex(
        columns=column_names, fill_value=0
    )

    # Scale before prediction
    customer_scaled = scaler.transform(customer_encoded)

    # Predict churn
    prediction = int(model.predict(customer_scaled)[0])

    classes = list(model.classes_)
    churn_index = classes.index(1)
    churn_probability = model.predict_proba(customer_scaled)[0][churn_index]

    st.subheader("Prediction Result")
    st.metric("Probability of Churn", f"{churn_probability:.1%}")

    if prediction == 1:
        st.error("Likely to churn")
    else:
        st.success("Likely to stay")
