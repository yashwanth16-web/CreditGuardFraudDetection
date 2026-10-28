
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "Class"

# Provisional candidate based on our validation experiments.
# Keep this choice fixed before looking at test results.
MODEL_NAME = "XGBoost"
MODEL_FILE = "xgboost.joblib"
THRESHOLD_REPORT = REPORT_DIR / "threshold_comparison.csv"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_path = DATA_DIR / "test.csv"
    model_path = MODEL_DIR / MODEL_FILE

    if not test_path.exists():
        raise FileNotFoundError(test_path)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if not THRESHOLD_REPORT.exists():
        raise FileNotFoundError(THRESHOLD_REPORT)

    # Retrieve the threshold selected on validation data.
    thresholds = pd.read_csv(THRESHOLD_REPORT)
    selected = thresholds.loc[
        thresholds["model"] == MODEL_NAME
    ]

    if selected.empty:
        raise ValueError(
            f"No validation threshold found for {MODEL_NAME}"
        )

    threshold = float(selected.iloc[0]["threshold"])

    # Load the untouched test partition.
    test = pd.read_csv(test_path)
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].astype(int)

    model = joblib.load(model_path)

    # XGBoost produces fraud probabilities.
    scores = model.predict_proba(X_test)[:, 1]
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": MODEL_NAME,
        "threshold": threshold,
        "test_rows": len(test),
        "fraud_cases": int(y_test.sum()),
        "average_precision": average_precision_score(
            y_test, scores
        ),
        "roc_auc": roc_auc_score(y_test, scores),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Model: {MODEL_NAME}")
    print(f"Validation-selected threshold: {threshold:.6f}")
    print(f"Test transactions: {len(test):,}")
    print(f"Test fraud cases: {int(y_test.sum()):,}")

    for metric in [
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1_score",
    ]:
        print(f"{metric}: {metrics[metric]:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    output_path = REPORT_DIR / "final_test_results.csv"
    pd.DataFrame([metrics]).to_csv(output_path, index=False)

    print(f"\nSaved final test results: {output_path}")
    print("Final evaluation completed.")


if __name__ == "__main__":
    main()
