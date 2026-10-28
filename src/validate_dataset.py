
from pathlib import Path

import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "creditcard.csv"
)


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Place creditcard.csv inside data/raw/")
        return

    df = pd.read_csv(DATA_PATH)

    print("\n--- DATASET VALIDATION REPORT ---")
    print(f"File: {DATA_PATH.name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")

    print("\n--- FIRST FIVE ROWS ---")
    print(df.head())

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum().sum())

    print("\n--- DUPLICATE ROWS ---")
    print(df.duplicated().sum())

    print("\n--- TARGET DISTRIBUTION ---")
    if "Class" not in df.columns:
        print("ERROR: Target column 'Class' is missing.")
        return

    print(df["Class"].value_counts().sort_index())

    print("\n--- TARGET PERCENTAGES ---")
    print(
        (df["Class"].value_counts(normalize=True).sort_index() * 100)
        .round(4)
    )

    print("\n--- VALIDATION COMPLETE ---")


if __name__ == "__main__":
    main()
