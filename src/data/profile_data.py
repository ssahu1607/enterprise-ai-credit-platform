import pandas as pd


DATA_PATH = "data/raw/loan_applications.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    print("\n========== SHAPE ==========")
    print(df.shape)

    print("\n========== COLUMNS ==========")
    print(df.columns.tolist())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATES ==========")
    print(df.duplicated().sum())

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["default"].value_counts())

    print("\n========== TARGET PERCENTAGE ==========")
    print(df["default"].value_counts(normalize=True) * 100)

    print("\n========== NUMERICAL SUMMARY ==========")
    print(df.describe())


if __name__ == "__main__":
    main()