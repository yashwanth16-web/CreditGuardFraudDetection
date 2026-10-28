
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
FIGURES_DIR = ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== SUMMARY STATISTICS ==========")
    print(df[["Time", "Amount"]].describe().round(3))

    # 1. Target distribution
    counts = df["Class"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    print("\n========== CLASS DISTRIBUTION ==========")
    print(f"Legitimate: {counts[0]:,}")
    print(f"Fraud: {counts[1]:,}")
    print(f"Fraud percentage: {df['Class'].mean() * 100:.4f}%")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        x=["Legitimate", "Fraud"],
        y=[counts[0], counts[1]],
        ax=ax,
    )
    ax.set_title("Legitimate vs Fraudulent Transactions")
    ax.set_xlabel("Transaction class")
    ax.set_ylabel("Number of transactions")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "class_distribution.png",
        dpi=150,
    )
    plt.close(fig)

    # 2. Transaction amount distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(
        data=df,
        x="Amount",
        bins=60,
        ax=axes[0],
    )
    axes[0].set_title("Transaction Amount Distribution")
    axes[0].set_xlim(left=0)

    sns.boxplot(
        data=df,
        x="Class",
        y="Amount",
        ax=axes[1],
    )
    axes[1].set_title("Transaction Amount by Class")
    axes[1].set_xticks([0, 1], ["Legitimate", "Fraud"])

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "amount_analysis.png",
        dpi=150,
    )
    plt.close(fig)

    # 3. Transaction time distribution
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="Time",
        hue="Class",
        bins=60,
        element="step",
        stat="density",
        common_norm=False,
        ax=ax,
    )

    ax.set_title("Transaction Time Distribution by Class")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "time_distribution.png",
        dpi=150,
    )
    plt.close(fig)

    # 4. Duplicate investigation
    print("\n========== DUPLICATE INVESTIGATION ==========")

    full_duplicates = df.duplicated().sum()

    feature_columns = [
        column for column in df.columns
        if column != "Class"
    ]

    duplicate_features = df.duplicated(
        subset=feature_columns,
        keep=False,
    )

    conflicting_groups = (
        df.loc[duplicate_features]
        .groupby(feature_columns, dropna=False)["Class"]
        .nunique()
    )

    conflicting_groups = conflicting_groups[
        conflicting_groups > 1
    ]

    print(f"Exact duplicate rows: {full_duplicates:,}")
    print(
        "Rows involved in repeated feature combinations:",
        f"{duplicate_features.sum():,}",
    )
    print(
        "Feature combinations with conflicting labels:",
        f"{len(conflicting_groups):,}",
    )

    # 5. Fraud rates by class and transaction amount
    print("\n========== AMOUNT SUMMARY BY CLASS ==========")

    amount_summary = df.groupby("Class")["Amount"].agg(
        ["count", "mean", "median", "max"]
    )

    print(amount_summary.round(3))

    print("\n========== SAVED VISUALIZATIONS ==========")
    for filename in [
        "class_distribution.png",
        "amount_analysis.png",
        "time_distribution.png",
    ]:
        print(FIGURES_DIR / filename)

    print("\nEDA completed successfully!")


if __name__ == "__main__":
    main()
