
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"

REPORT_FILES = [
    "logistic_regression_validation.csv",
    "decision_tree_validation.csv",
    "random_forest_validation.csv",
    "xgboost_validation.csv",
    "linear_svm_validation.csv",
    "isolation_forest_validation.csv",
]

REQUIRED_METRICS = [
    "average_precision",
    "roc_auc",
    "precision",
    "recall",
    "f1_score",
]


def main():
    results = []

    print("\n========== MODEL COMPARISON ==========")

    for filename in REPORT_FILES:
        path = REPORT_DIR / filename

        if not path.exists():
            print(f"Missing report: {filename}")
            continue

        df = pd.read_csv(path)

        if df.empty:
            print(f"Empty report: {filename}")
            continue

        row = df.iloc[0].to_dict()

        missing = [
            metric for metric in REQUIRED_METRICS
            if metric not in row or pd.isna(row[metric])
        ]

        if missing:
            print(
                f"Skipping {filename}; missing metrics: {missing}"
            )
            continue

        # Normalize the threshold field name across models.
        threshold = row.get(
            "threshold",
            row.get(
                "decision_threshold",
                row.get("threshold_method", "model default"),
            ),
        )

        results.append({
            "model": row.get("model", filename),
            "average_precision": float(row["average_precision"]),
            "roc_auc": float(row["roc_auc"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1_score": float(row["f1_score"]),
            "threshold": threshold,
            "source_report": filename,
        })

    if not results:
        raise RuntimeError(
            "No valid model reports found. Train the models first."
        )

    comparison = pd.DataFrame(results)

    # Average precision is the primary ranking metric for this
    # heavily imbalanced dataset. This is only a preliminary ranking.
    comparison = comparison.sort_values(
        by="average_precision",
        ascending=False,
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "preliminary_rank",
        range(1, len(comparison) + 1),
    )

    output_path = REPORT_DIR / "model_comparison.csv"
    comparison.to_csv(output_path, index=False)

    display_columns = [
        "preliminary_rank",
        "model",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1_score",
    ]

    print("\n--- VALIDATION LEADERBOARD ---")
    print(
        comparison[display_columns].to_string(
            index=False,
            formatters={
                "average_precision": "{:.4f}".format,
                "roc_auc": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1_score": "{:.4f}".format,
            },
        )
    )

    print(f"\nSaved comparison report: {output_path}")
    print(
        "\nPreliminary ranking only: thresholds and model "
        "configurations are not yet fully standardized."
    )


if __name__ == "__main__":
    main()
