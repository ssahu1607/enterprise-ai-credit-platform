import os
import pandas as pd


LOAN_PATH = "data/processed/loan_applications_clean.csv"
TRANSACTION_PATH = "data/processed/transactions_clean.csv"

OUTPUT_PATH = "data/features/ml_features.csv"


def create_transaction_features(transactions):

    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"]
    )

    # Basic customer-level transaction statistics
    transaction_features = (
        transactions
        .groupby("customer_id")
        .agg(
            total_transaction_amount=(
                "amount",
                "sum"
            ),
            avg_transaction_amount=(
                "amount",
                "mean"
            ),
            transaction_count=(
                "amount",
                "count"
            ),
            unique_merchant_categories=(
                "merchant_category",
                "nunique"
            )
        )
        .reset_index()
    )

    # Debit transactions
    debit = transactions[
        transactions["transaction_type"] == "Debit"
    ]

    debit_features = (
        debit
        .groupby("customer_id")
        .agg(
            total_debit_amount=(
                "amount",
                "sum"
            ),
            avg_debit_amount=(
                "amount",
                "mean"
            ),
            debit_transaction_count=(
                "amount",
                "count"
            )
        )
        .reset_index()
    )

    # Credit transactions
    credit = transactions[
        transactions["transaction_type"] == "Credit"
    ]

    credit_features = (
        credit
        .groupby("customer_id")
        .agg(
            total_credit_amount=(
                "amount",
                "sum"
            ),
            avg_credit_amount=(
                "amount",
                "mean"
            ),
            credit_transaction_count=(
                "amount",
                "count"
            )
        )
        .reset_index()
    )

    return (
        transaction_features
        .merge(
            debit_features,
            on="customer_id",
            how="left"
        )
        .merge(
            credit_features,
            on="customer_id",
            how="left"
        )
    )


def create_final_features():

    loans = pd.read_csv(LOAN_PATH)
    transactions = pd.read_csv(TRANSACTION_PATH)

    transaction_features = create_transaction_features(
        transactions
    )

    # Combine loan and transaction features
    final_df = loans.merge(
        transaction_features,
        on="customer_id",
        how="left"
    )

    # Avoid division by zero
    final_df["debit_credit_ratio"] = (
        final_df["total_debit_amount"]
        /
        final_df["total_credit_amount"].replace(
            0,
            1
        )
    )

    # Fill missing transaction-derived values
    transaction_columns = [
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "unique_merchant_categories",
        "total_debit_amount",
        "avg_debit_amount",
        "debit_transaction_count",
        "total_credit_amount",
        "avg_credit_amount",
        "credit_transaction_count",
        "debit_credit_ratio"
    ]

    for column in transaction_columns:
        final_df[column] = final_df[column].fillna(0)

    return final_df


def main():

    os.makedirs(
        "data/features",
        exist_ok=True
    )

    final_df = create_final_features()

    final_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "Feature engineering completed."
    )

    print(
        "Final dataset shape:",
        final_df.shape
    )

    print(
        "\nFeature columns:"
    )

    print(
        final_df.columns.tolist()
    )


if __name__ == "__main__":
    main()