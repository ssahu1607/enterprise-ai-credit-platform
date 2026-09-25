import os
import numpy as np
import pandas as pd


# Reproducibility
np.random.seed(42)

# Number of customers/applications
N = 10000


def generate_loan_applications(n=10000):

    customer_id = [
        f"CUST_{i:06d}"
        for i in range(1, n + 1)
    ]

    age = np.random.randint(21, 65, n)

    annual_income = np.random.lognormal(
        mean=10.8,
        sigma=0.5,
        size=n
    )

    annual_income = np.clip(
        annual_income,
        180000,
        5000000
    )

    loan_amount = np.random.lognormal(
        mean=12.0,
        sigma=0.6,
        size=n
    )

    loan_amount = np.clip(
        loan_amount,
        50000,
        5000000
    )

    credit_score = np.random.normal(
        700,
        70,
        n
    )

    credit_score = np.clip(
        credit_score,
        300,
        850
    )

    employment_years = np.random.randint(
        0,
        31,
        n
    )

    existing_loans = np.random.poisson(
        1.5,
        n
    )

    previous_defaults = np.random.binomial(
        2,
        0.08,
        n
    )

    debt_to_income = np.random.beta(
        2,
        5,
        n
    )

    dependents = np.random.randint(
        0,
        5,
        n
    )

    account_age_months = np.random.randint(
        6,
        240,
        n
    )

    employment_type = np.random.choice(
        [
            "Salaried",
            "Self_Employed",
            "Business",
            "Contract"
        ],
        n,
        p=[0.55, 0.20, 0.15, 0.10]
    )

    home_ownership = np.random.choice(
        [
            "Owned",
            "Rented",
            "Mortgage"
        ],
        n,
        p=[0.35, 0.45, 0.20]
    )

    loan_purpose = np.random.choice(
        [
            "Home",
            "Education",
            "Medical",
            "Vehicle",
            "Personal",
            "Business"
        ],
        n
    )

    monthly_expenses = (
        annual_income / 12
    ) * np.random.uniform(
        0.25,
        0.75,
        n
    )

    # Risk score used ONLY to generate the synthetic target.
    # It will NOT be included as a model feature.
    risk_score = (
        -0.004 * (credit_score - 650)
        + 3.0 * debt_to_income
        + 0.8 * previous_defaults
        + 0.15 * existing_loans
        + 0.00000015 * loan_amount
        - 0.00000010 * annual_income
    )

    probability_of_default = (
        1 / (1 + np.exp(-risk_score))
    )

    default = np.random.binomial(
        1,
        probability_of_default
    )

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "annual_income": annual_income.round(2),
        "loan_amount": loan_amount.round(2),
        "credit_score": credit_score.round(0),
        "employment_years": employment_years,
        "existing_loans": existing_loans,
        "previous_defaults": previous_defaults,
        "debt_to_income": debt_to_income.round(3),
        "dependents": dependents,
        "account_age_months": account_age_months,
        "employment_type": employment_type,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "monthly_expenses": monthly_expenses.round(2),
        "default": default
    })

    return df


def generate_transactions(customers, transactions_per_customer=10):

    rows = []

    transaction_categories = [
        "Groceries",
        "Utilities",
        "Travel",
        "Shopping",
        "Healthcare",
        "Education",
        "Dining",
        "Entertainment"
    ]

    for customer_id in customers:

        for _ in range(transactions_per_customer):

            amount = np.random.lognormal(
                mean=5,
                sigma=1
            )

            transaction_type = np.random.choice(
                ["Debit", "Credit"],
                p=[0.75, 0.25]
            )

            rows.append({
                "customer_id": customer_id,
                "transaction_date": pd.Timestamp(
                    "2026-01-01"
                ) + pd.to_timedelta(
                    np.random.randint(0, 270),
                    unit="D"
                ),
                "transaction_type": transaction_type,
                "amount": round(amount, 2),
                "merchant_category": np.random.choice(
                    transaction_categories
                )
            })

    return pd.DataFrame(rows)


def main():

    os.makedirs("data/raw", exist_ok=True)

    # Loan application dataset
    loans = generate_loan_applications(N)

    loans.to_csv(
        "data/raw/loan_applications.csv",
        index=False
    )

    # Transaction dataset
    transactions = generate_transactions(
        loans["customer_id"],
        transactions_per_customer=10
    )

    transactions.to_csv(
        "data/raw/transactions.csv",
        index=False
    )

    print("Dataset generation completed.")
    print(f"Loan applications: {len(loans)}")
    print(f"Transactions: {len(transactions)}")


if __name__ == "__main__":
    main()