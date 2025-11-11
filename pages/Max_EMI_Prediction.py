import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os


MODEL_DIR = "models"

emi_model = joblib.load(os.path.join(MODEL_DIR, "best_max_emi_model.pkl"))
emi_scaler = joblib.load(os.path.join(MODEL_DIR, "emi_scaler.pkl"))
emi_features = joblib.load(os.path.join(MODEL_DIR, "emi_features.pkl"))


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
    }


st.title("💰 Maximum EMI Predictor")

st.write("Enter customer details below to calculate their maximum affordable EMI:")

monthly_salary = st.number_input("Monthly Salary", min_value=1000.0)
monthly_expenses = st.number_input("Total Monthly Expenses", min_value=0.0)
current_emi = st.number_input("Current EMI", min_value=0.0)
credit_score = st.number_input("Credit Score (300–900)", min_value=300.0, max_value=900.0)
bank_balance = st.number_input("Bank Balance", min_value=0.0)
req_amount = st.number_input("Requested Loan Amount", min_value=0.0)
req_tenure = st.number_input("Requested Tenure (months)", min_value=1)


if st.button("🔍 Predict Max EMI"):
    raw = {
        "monthly_salary": monthly_salary,
        "total_expenses": monthly_expenses,
        "current_emi_amount": current_emi,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "requested_amount": req_amount,
        "requested_tenure": req_tenure,
    }

    engineered = engineer_features(raw)

    df = pd.DataFrame([engineered])[emi_features]
    scaled = emi_scaler.transform(df)
    pred = emi_model.predict(scaled)[0]

    st.subheader("Result")
    st.write(f"### Maximum Affordable EMI: **₹{pred:,.2f}**")
