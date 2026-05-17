
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("classification_dataset.csv")

X = df.drop("Approved", axis=1)
y = df["Approved"]

scaler = StandardScaler()
scaler.fit(X)

# Load Trained Model
with open("logistic_classification_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Logistic Classification App", layout="centered")

st.title("Logistic Classification Prediction App")

st.write("Enter customer details to predict loan approval status.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", value=50000.0)
credit_score = st.number_input("Credit Score", value=650)
loan_amount = st.number_input("Loan Amount", value=100000.0)
experience = st.number_input("Experience", value=5)
savings = st.number_input("Savings", value=50000.0)
transaction_count = st.number_input("Transaction Count", value=50)

if st.button("Predict"):

    input_data = pd.DataFrame([[
        age,
        income,
        credit_score,
        loan_amount,
        experience,
        savings,
        transaction_count
    ]], columns=X.columns)

    input_scaled = scaler.transform(input_data)

  
    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")

    st.write(f"Approval Probability: {probability[1]:.2f}")
    st.write(f"Rejection Probability: {probability[0]:.2f}")

st.markdown("---")
st.write("Built using Streamlit and Logistic Regression")
