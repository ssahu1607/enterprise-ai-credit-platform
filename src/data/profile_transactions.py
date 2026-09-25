import pandas as pd


DATA_PATH = "data/raw/transactions.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    print("\n========== SHAPE ==========")
    print(df.shape)

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATES ==========")
    print(df.duplicated().sum())

    print("\n========== TRANSACTION TYPES ==========")
    print(df["transaction_type"].value_counts())

    print("\n========== MERCHANT CATEGORIES ==========")
    print(df["merchant_category"].value_counts())

    print("\n========== NUMERICAL SUMMARY ==========")
    print(df["amount"].describe())


if __name__ == "__main__":
    main()