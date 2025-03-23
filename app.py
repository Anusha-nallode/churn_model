import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("customer_churn_model.pkl")

# Define expected feature names (must match training data)
expected_features = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

# Streamlit UI
st.title("Bank Customer Churn Prediction")

# User input form
st.sidebar.header("Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, step=1)
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=150.0, step=0.1)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=9000.0, step=0.1)

# Convert input into a DataFrame
user_input = pd.DataFrame({
    "SeniorCitizen": [SeniorCitizen],
    "tenure": [tenure],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
})

# One-hot encode categorical features
categorical_mapping = {
    "gender": f"gender_{gender}",
    "Partner": f"Partner_{Partner}",
    "Dependents": f"Dependents_{Dependents}",
    "PhoneService": f"PhoneService_{PhoneService}",
    "MultipleLines": f"MultipleLines_{MultipleLines}",
    "InternetService": f"InternetService_{InternetService}",
    "OnlineSecurity": f"OnlineSecurity_{OnlineSecurity}",
    "OnlineBackup": f"OnlineBackup_{OnlineBackup}",
    "DeviceProtection": f"DeviceProtection_{DeviceProtection}",
    "TechSupport": f"TechSupport_{TechSupport}",
    "StreamingTV": f"StreamingTV_{StreamingTV}",
    "StreamingMovies": f"StreamingMovies_{StreamingMovies}",
    "Contract": f"Contract_{Contract}",
    "PaperlessBilling": f"PaperlessBilling_{PaperlessBilling}",
    "PaymentMethod": f"PaymentMethod_{PaymentMethod}",
}

# Add one-hot encoded categorical values
for key, value in categorical_mapping.items():
    for feature in expected_features:
        user_input[feature] = 1 if feature == value else 0

# Print user input for debugging
print("User input columns:", user_input.columns.tolist())

# Ensure all features match training
missing_features = [f for f in expected_features if f not in user_input.columns]

if missing_features:
    st.error(f"Missing features: {missing_features}")
else:
    prediction = model.predict(user_input)
    churn_probability = model.predict_proba(user_input)[0][1]  # Get probability of churn

    st.subheader("Churn Prediction Result")
    st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    st.write(f"Churn Probability: {churn_probability:.2%}")
