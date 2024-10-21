import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Function to preprocess input data
def preprocess_input(employment_status, education_level, marital_status, home_ownership_status, loan_purpose, loan_amount, loan_term, credit_score, income):
    le = LabelEncoder()
    employment_status = le.fit_transform([employment_status])[0]
    education_level = le.fit_transform([education_level])[0]
    marital_status = le.fit_transform([marital_status])[0]
    home_ownership_status = le.fit_transform([home_ownership_status])[0]
    loan_purpose = le.fit_transform([loan_purpose])[0]
    return [employment_status, education_level, marital_status, home_ownership_status, loan_purpose, loan_amount, loan_term, credit_score, income]


# Streamlit app
st.title("Loan Approval Prediction")

# Input fields
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed"])
education_level = st.selectbox("Education Level", ["Graduate", "High School", "Undergraduate"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
home_ownership_status = st.selectbox("Home Ownership Status", ["Own", "Rent", "Mortgage"])
loan_purpose = st.selectbox("Loan Purpose", ["Home Improvement", "Debt Consolidation", "Business", "Personal"])
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0)
credit_score = st.number_input("Credit Score", min_value=0)
income = st.number_input("Income", min_value=0)

# Make prediction when the user clicks the button
if st.button("Predict Loan Approval"):
    input_data = preprocess_input(employment_status, education_level, marital_status, home_ownership_status, loan_purpose, loan_amount, loan_term, credit_score, income)

    prediction = model.predict([input_data])[0]

    if prediction == 1:
        st.success("Loan is likely to be approved.")
    else:
        st.error("Loan is likely to be denied.")
