
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

TARGET = "Class"
TIME_COLUMN = "Time"


def main():
    split_names = ["train", "validation", "test"]
    splits = {}

    print("\n========== SPLIT QUALITY REPORT ==========")

    for name in split_names:
        path = DATA_DIR / f"{name}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing split file: {path}"
            )

        df = pd.read_csv(path)

        if TARGET not in df.columns or TIME_COLUMN not in df.columns:
            raise ValueError(
                f"{name}.csv is missing required columns."
            )

        if df.empty:
            raise ValueError(f"{name}.csv is empty.")

        if df[TARGET].isna().any():
            raise ValueError(
                f"{name}.csv contains missing target values."
            )

        if not df[TARGET].isin([0, 1]).all():
            raise ValueError(
                f"{name}.csv has unexpected target labels."
            )

        splits[name] = df

        fraud = int((df[TARGET] == 1).sum())
        legitimate = int((df[TARGET] == 0).sum())

        print(f"\n--- {name.upper()} ---")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print(f"Legitimate: {legitimate:,}")
        print(f"Fraud: {fraud:,}")
        print(f"Fraud rate: {df[TARGET].mean() * 100:.4f}%")
        print(
            f"Time range: {df[TIME_COLUMN].min():.2f}"
            f" to {df[TIME_COLUMN].max():.2f}"
        )
        print(f"Missing values: {df.isna().sum().sum():,}")
        print(f"Exact duplicate rows: {df.duplicated().sum():,}")

    # Confirm that every split uses the same columns.
    reference_columns = list(splits["train"].columns)

    for name, df in splits.items():
        if list(df.columns) != reference_columns:
            raise ValueError(
                f"Column mismatch in {name}.csv"
            )

    # Check chronological separation.
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    assert train[TIME_COLUMN].max() <= validation[TIME_COLUMN].min()
    assert validation[TIME_COLUMN].max() <= test[TIME_COLUMN].min()

    # Check for exact records shared across different partitions.
    def row_hashes(df):
        return set(
            pd.util.hash_pandas_object(
                df[reference_columns],
                index=False,
            ).tolist()
        )

    train_hashes = row_hashes(train)
    validation_hashes = row_hashes(validation)
    test_hashes = row_hashes(test)

    overlaps = {
        "train-validation": len(train_hashes & validation_hashes),
        "train-test": len(train_hashes & test_hashes),
        "validation-test": len(validation_hashes & test_hashes),
    }

    print("\n========== CROSS-SPLIT CHECKS ==========")

    for pair, count in overlaps.items():
        print(f"Exact shared records ({pair}): {count}")

    total_rows = sum(len(df) for df in splits.values())

    print(f"\nTotal rows across splits: {total_rows:,}")
    print("Chronological ordering: PASSED")
    print("Column consistency: PASSED")
    print("Target validation: PASSED")

    print("\nSplit validation completed!")


if __name__ == "__main__":
    main()
