import streamlit as st
import pandas as pd
import pickle
import joblib

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = joblib.load(f)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load the encoders
try:
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    st.success("Encoders loaded successfully.")
except Exception as e:
    st.error(f"Error loading encoders: {e}")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

# Convert input to DataFrame
input_data = pd.DataFrame([{ 
    "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner, "Dependents": Dependents, "tenure": tenure, 
    "PhoneService": PhoneService, "MultipleLines": MultipleLines, "InternetService": InternetService, "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection, "TechSupport": TechSupport, "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies, "Contract": Contract, "PaperlessBilling": PaperlessBilling, "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges
}])

# Handle missing or unknown values
if TotalCharges == 0:
    input_data["TotalCharges"] = MonthlyCharges * tenure  # Approximate missing TotalCharges

# Encoding step with debugging
for column in input_data.columns:
    if column in encoders:
        try:
            st.write(f"Encoding column: {column}")
            input_data[column] = encoders[column].transform(input_data[column])
        except Exception as e:
            st.warning(f"Could not encode {column}: {e}. Filling with default.")
            input_data[column] = 0  # Assign default numeric value if encoding fails

# Ensure columns match model expectations
missing_cols = [col for col in model.feature_names_in_ if col not in input_data.columns]
for col in missing_cols:
    st.warning(f"Adding missing column {col} with default value 0.")
    input_data[col] = 0  # Add missing columns with default values

# Prediction step with debugging
if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_data)
        pred_prob = model.predict_proba(input_data)[0][1]  # Probability of churn
        result = "Churn" if prediction[0] == 1 else "No Churn"
        st.write(f"### Prediction: {result}")
        st.write(f"### Churn Probability: {pred_prob:.2%}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
