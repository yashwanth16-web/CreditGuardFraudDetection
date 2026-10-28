
from pathlib import Path

import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "Class"
TIME_COLUMN = "Time"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load original data without modifying the source file.
    df = pd.read_csv(DATA_PATH)

    required_columns = {TARGET, TIME_COLUMN}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1.")

    if df[TIME_COLUMN].isna().any():
        raise ValueError("Time column contains missing values.")

    # Sort chronologically. Stable sorting preserves the original
    # order of rows with equal timestamps.
    df = df.sort_values(
        TIME_COLUMN,
        kind="stable",
    ).reset_index(drop=True)

    # Calculate approximate time-based cutoffs.
    cutoff_1 = df[TIME_COLUMN].quantile(0.70)
    cutoff_2 = df[TIME_COLUMN].quantile(0.85)

    # Keep equal timestamps together to avoid splitting transactions
    # with the same recorded time across different partitions.
    train = df[df[TIME_COLUMN] <= cutoff_1].copy()

    validation = df[
        (df[TIME_COLUMN] > cutoff_1)
        & (df[TIME_COLUMN] <= cutoff_2)
    ].copy()

    test = df[df[TIME_COLUMN] > cutoff_2].copy()

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    print("\n========== CHRONOLOGICAL DATA SPLIT ==========")
    print(f"Original rows: {len(df):,}")
    print(f"First time cutoff: {cutoff_1:.2f}")
    print(f"Second time cutoff: {cutoff_2:.2f}")

    for name, part in splits.items():
        fraud_count = int((part[TARGET] == 1).sum())
        legitimate_count = int((part[TARGET] == 0).sum())
        fraud_rate = part[TARGET].mean() * 100

        print(f"\n--- {name.upper()} ---")
        print(f"Rows: {len(part):,}")
        print(f"Legitimate: {legitimate_count:,}")
        print(f"Fraud: {fraud_count:,}")
        print(f"Fraud percentage: {fraud_rate:.4f}%")

        if part.empty:
            raise ValueError(
                f"{name} split is empty. Review the time cutoffs."
            )

        output_path = PROCESSED_DIR / f"{name}.csv"
        part.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    # Confirm that the partitions cover the original rows.
    total_split_rows = sum(len(part) for part in splits.values())

    assert total_split_rows == len(df)
    assert train[TIME_COLUMN].max() <= validation[TIME_COLUMN].min()
    assert validation[TIME_COLUMN].max() <= test[TIME_COLUMN].min()

    print("\n========== SPLIT VALIDATION ==========")
    print("All rows are accounted for.")
    print("Chronological order is preserved.")
    print("Equal timestamps are kept within one partition.")
    print("\nData preparation completed successfully!")


if __name__ == "__main__":
    main()
