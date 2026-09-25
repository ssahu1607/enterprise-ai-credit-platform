import os
import pandas as pd


LOAN_PATH = "data/raw/loan_applications.csv"
TRANSACTION_PATH = "data/raw/transactions.csv"

OUTPUT_DIR = "data/processed"


def clean_loan_data(df):

    # Remove duplicate records
    df = df.drop_duplicates()

    # Basic validation
    df = df[
        (df["age"] >= 18) &
        (df["credit_score"] >= 300) &
        (df["credit_score"] <= 850) &
        (df["loan_amount"] > 0) &
        (df["annual_income"] > 0)
    ]

    # Handle missing numerical values
    numerical_columns = [
        "annual_income",
        "loan_amount",
        "credit_score",
        "employment_years",
        "existing_loans",
        "previous_defaults",
        "debt_to_income",
        "dependents",
        "account_age_months",
        "monthly_expenses"
    ]

    for column in numerical_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    # Handle missing categorical values
    categorical_columns = [
        "employment_type",
        "home_ownership",
        "loan_purpose"
    ]

    for column in categorical_columns:
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

    return df


def clean_transaction_data(df):

    df = df.drop_duplicates()

    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"]
    )

    df = df[df["amount"] > 0]

    df["merchant_category"] = df[
        "merchant_category"
    ].fillna("Unknown")

    df["transaction_type"] = df[
        "transaction_type"
    ].fillna("Unknown")

    return df


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    loans = pd.read_csv(LOAN_PATH)
    transactions = pd.read_csv(TRANSACTION_PATH)

    print("Before cleaning:")
    print("Loans:", loans.shape)
    print("Transactions:", transactions.shape)

    loans_clean = clean_loan_data(loans)
    transactions_clean = clean_transaction_data(
        transactions
    )

    loans_clean.to_csv(
        f"{OUTPUT_DIR}/loan_applications_clean.csv",
        index=False
    )

    transactions_clean.to_csv(
        f"{OUTPUT_DIR}/transactions_clean.csv",
        index=False
    )

    print("\nAfter cleaning:")
    print("Loans:", loans_clean.shape)
    print("Transactions:", transactions_clean.shape)

    print("\nCleaning completed successfully.")


if __name__ == "__main__":
    main()