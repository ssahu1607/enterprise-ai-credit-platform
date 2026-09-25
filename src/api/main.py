import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# 1. LOAD MODEL
# ============================================================

MODEL_PATH = "models/credit_risk_model.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# 2. CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Enterprise Credit Risk API",
    description="API for credit default risk prediction",
    version="1.0"
)


# ============================================================
# 3. INPUT SCHEMA
# ============================================================

class LoanApplication(BaseModel):

    age: int
    annual_income: float
    loan_amount: float
    credit_score: int
    employment_years: float
    existing_loans: int
    previous_defaults: int
    debt_to_income: float
    dependents: int
    account_age_months: int
    employment_type: str
    home_ownership: str
    loan_purpose: str
    monthly_expenses: float

    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: int
    unique_merchant_categories: int
    total_debit_amount: float
    avg_debit_amount: float
    debit_transaction_count: int
    total_credit_amount: float
    avg_credit_amount: float
    credit_transaction_count: int
    debit_credit_ratio: float


# ============================================================
# 4. HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "credit-risk-model"
    }


# ============================================================
# 5. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(application: LoanApplication):

    # Convert request to DataFrame
    input_data = pd.DataFrame(
        [application.model_dump()]
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability of default
    probability = model.predict_proba(input_data)[0][1]

    # Business decision
    if probability >= 0.50:
        decision = "HIGH_RISK"
    else:
        decision = "LOW_RISK"

    return {
        "prediction": int(prediction),
        "default_probability": round(float(probability), 4),
        "decision": decision
    }