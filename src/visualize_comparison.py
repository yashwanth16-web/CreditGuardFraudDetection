
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

INPUT_PATH = REPORT_DIR / "threshold_comparison.csv"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing comparison report: {INPUT_PATH}"
        )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required = [
        "model",
        "precision",
        "recall",
        "f1_score",
        "average_precision",
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values("f1_score", ascending=True)

    # Save a presentation-friendly copy of the metrics.
    summary_path = REPORT_DIR / "model_comparison_final.csv"
    df.to_csv(summary_path, index=False)

    # Compare precision, recall, and F1 at tuned thresholds.
    ax = df.set_index("model")[
        ["precision", "recall", "f1_score"]
    ].plot(
        kind="barh",
        figsize=(11, 7),
        xlim=(0, 1),
        width=0.8,
    )

    ax.set_title("Fraud Detection Model Comparison")
    ax.set_xlabel("Score")
    ax.set_ylabel("Model")
    ax.legend(
        ["Precision", "Recall", "F1-score"],
        loc="lower right",
    )

    plt.tight_layout()

    chart_path = FIGURE_DIR / "model_comparison.png"
    plt.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close()

    # Average precision is useful for ranking rare fraud cases.
    ap_df = df.sort_values("average_precision", ascending=True)

    ax = ap_df.set_index("model")["average_precision"].plot(
        kind="barh",
        figsize=(10, 6),
        xlim=(0, 1),
        legend=False,
    )

    ax.set_title("Average Precision by Model")
    ax.set_xlabel("Average precision")
    ax.set_ylabel("Model")

    plt.tight_layout()

    ap_chart_path = FIGURE_DIR / "average_precision_comparison.png"
    plt.savefig(ap_chart_path, dpi=160, bbox_inches="tight")
    plt.close()

    print("\n========== COMPARISON FILES CREATED ==========")
    print(f"Summary CSV: {summary_path}")
    print(f"Metrics chart: {chart_path}")
    print(f"Average precision chart: {ap_chart_path}")
    print("\nVisual comparison completed!")


if __name__ == "__main__":
    main()
