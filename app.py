import os
import joblib
import pandas as pd
import streamlit as st

from model_downloader import download_models
download_models()

st.set_page_config(page_title="EMI Risk Assessment", page_icon="🏦", layout="centered")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------- Utilities
def safe_div(a, b):
    try:
        b = float(b)
        return float(a) / b if b != 0 else 0.0
    except:
        return 0.0

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

def _safe_load_pickle(basename: str):
    """
    Try load → if fails, force redownload that file and try again.
    Shows a Streamlit error if still failing.
    """
    path = ensure_one(basename, force=False)
    try:
        return joblib.load(path)
    except Exception as e1:
        # force re-download and try again
        path = ensure_one(basename, force=True)
        try:
            return joblib.load(path)
        except Exception as e2:
            st.error(
                f"Failed to load '{basename}'. "
                "Please try 'Refresh models' below or check Drive link permissions."
            )
            raise e2

# -------- Sidebar: maintenance
st.sidebar.header("Maintenance")
if st.sidebar.button("🔄 Refresh models (force re-download)"):
    download_all(force=True)
    st.sidebar.success("Models refreshed. Reload the page.")

with st.spinner("Preparing models..."):
    download_all(force=False)

# Load artifacts safely
elig_model        = _safe_load_pickle("best_eligibility_model.pkl")
elig_scaler       = _safe_load_pickle("eligibility_scaler.pkl")
elig_features     = _safe_load_pickle("eligibility_features.pkl")
emi_model         = _safe_load_pickle("best_max_emi_model.pkl")
emi_scaler        = _safe_load_pickle("emi_scaler.pkl")
emi_features      = _safe_load_pickle("emi_features.pkl")

# -------- UI
st.title("🏦 EMI Eligibility & Max EMI Prediction")

mode = st.radio("Choose task:", ["EMI Eligibility", "Max EMI Amount"])

st.subheader("📋 Enter Details")
cols1 = st.columns(3)
monthly_salary     = cols1[0].number_input("Monthly Salary", min_value=1000.0, step=100.0)
credit_score       = cols1[1].number_input("Credit Score (300–900)", min_value=300.0, max_value=900.0, step=1.0)
bank_balance       = cols1[2].number_input("Bank Balance", min_value=0.0, step=1000.0)

cols2 = st.columns(3)
years_emp          = cols2[0].number_input("Years of Employment", min_value=0.0, step=0.1)
requested_amount   = cols2[1].number_input("Requested Loan Amount", min_value=0.0, step=1000.0)
requested_tenure   = cols2[2].number_input("Requested Tenure (months)", min_value=1, step=1)

cols3 = st.columns(3)
monthly_rent       = cols3[0].number_input("Monthly Rent", min_value=0.0, step=100.0)
school_fees        = cols3[1].number_input("School Fees", min_value=0.0, step=100.0)
college_fees       = cols3[2].number_input("College Fees", min_value=0.0, step=100.0)

cols4 = st.columns(3)
travel_expenses    = cols4[0].number_input("Travel Expenses", min_value=0.0, step=100.0)
groceries          = cols4[1].number_input("Groceries & Utilities", min_value=0.0, step=100.0)
other_exp          = cols4[2].number_input("Other Monthly Expenses", min_value=0.0, step=100.0)

cols5 = st.columns(3)
current_emi        = cols5[0].number_input("Current EMI", min_value=0.0, step=100.0)
family_size        = cols5[1].number_input("Family Size", min_value=1, step=1)
dependents         = cols5[2].number_input("Dependents", min_value=0, step=1)

if st.button("🔍 Predict"):
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
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "family_size": family_size,
        "dependents": dependents,
    }

    feats = engineer_features(raw)

    if mode == "EMI Eligibility":
        X = pd.DataFrame([feats])[elig_features]
        Xs = elig_scaler.transform(X)
        y = elig_model.predict(Xs)[0]
        st.success("✅ Eligible" if y == 1 else "❌ Not Eligible")

    else:
        X = pd.DataFrame([feats])[emi_features]
        Xs = emi_scaler.transform(X)
        y = float(emi_model.predict(Xs)[0])
        st.success(f"✅ Maximum Affordable EMI: **₹{y:,.2f}**")

st.caption("Tip: if you see a model load error, use **Refresh models** in the sidebar.")
