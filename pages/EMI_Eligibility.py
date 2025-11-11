import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os


MODEL_DIR = "models"

elig_model = joblib.load(os.path.join(MODEL_DIR, "best_eligibility_model.pkl"))
elig_scaler = joblib.load(os.path.join(MODEL_DIR, "eligibility_scaler.pkl"))
elig_features = joblib.load(os.path.join(MODEL_DIR, "eligibility_features.pkl"))


def safe_div(a, b):
    try:
        return float(a) / float(b) if float(b) != 0 else 0
    except:
        return 0


def engineer_features(raw):
    return {
        "monthly_salary": float(raw["monthly_salary"]),
        "total_expenses": float(raw["total_expenses"]),
        "monthly_savings": float(raw["monthly_salary"]) - float(raw["total_expenses"]),
        "expense_ratio": safe_div(raw["total_expenses"], raw["monthly_salary"]),
        "emi_salary_ratio": safe_div(raw["current_emi_amount"], raw["monthly_salary"]),
        "balance_salary_ratio": safe_div(raw["bank_balance"], raw["monthly_salary"]),
        "credit_score": float(raw["credit_score"]),
        "requested_amount": float(raw["requested_amount"]),
        "requested_tenure": float(raw["requested_tenure"]),
        "bank_balance": float(raw["bank_balance"]),
        "years_of_employment": float(raw["years_of_employment"]),
        "dependents_ratio": safe_div(raw["dependents"], raw["family_size"]),
    }


st.title("✅ EMI Eligibility Checker")

st.write("Fill the following details to check if the customer is eligible for a loan:")

monthly_salary = st.number_input("Monthly Salary", min_value=1000.0)
monthly_rent = st.number_input("Monthly Rent", min_value=0.0)
school_fees = st.number_input("School Fees", min_value=0.0)
college_fees = st.number_input("College Fees", min_value=0.0)
travel_expenses = st.number_input("Travel Expenses", min_value=0.0)
groceries = st.number_input("Groceries & Utilities", min_value=0.0)
other_exp = st.number_input("Other Expenses", min_value=0.0)
current_emi = st.number_input("Current EMI", min_value=0.0)
credit_score = st.number_input("Credit Score (300–900)", min_value=300.0, max_value=900.0)
bank_balance = st.number_input("Bank Balance", min_value=0.0)
years_emp = st.number_input("Years of Employment", min_value=0.0)
family_size = st.number_input("Family Size", min_value=1)
dependents = st.number_input("Dependents", min_value=0)
req_amount = st.number_input("Requested Loan Amount", min_value=0.0)
req_tenure = st.number_input("Requested Tenure (months)", min_value=1)


if st.button("🔍 Check Eligibility"):
    total_expenses = (
        monthly_rent + school_fees + college_fees +
        travel_expenses + groceries + other_exp + current_emi
    )

    raw = {
        "monthly_salary": monthly_salary,
        "total_expenses": total_expenses,
        "current_emi_amount": current_emi,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "years_of_employment": years_emp,
        "requested_amount": req_amount,
        "requested_tenure": req_tenure,
        "family_size": family_size,
        "dependents": dependents,
    }

    engineered = engineer_features(raw)

    df = pd.DataFrame([engineered])[elig_features]
    scaled = elig_scaler.transform(df)
    pred = elig_model.predict(scaled)[0]

    status = "✅ Eligible" if pred == 1 else "❌ Not Eligible"

    st.subheader("Result")
    st.write(f"### EMI Eligibility: {status}")
