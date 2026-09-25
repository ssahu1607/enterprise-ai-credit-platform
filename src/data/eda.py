import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/features/ml_features.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    print("\n========== SHAPE ==========")
    print(df.shape)

    print("\n========== TARGET ==========")
    print(df["default"].value_counts())

    print("\n========== TARGET % ==========")
    print(
        df["default"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\n========== MISSING VALUES ==========")
    print(
        df.isnull().sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\n========== NUMERICAL SUMMARY ==========")
    print(df.describe().T)

    # Target distribution
    df["default"].value_counts().plot(
        kind="bar"
    )

    plt.title("Default Distribution")
    plt.xlabel("Default")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()