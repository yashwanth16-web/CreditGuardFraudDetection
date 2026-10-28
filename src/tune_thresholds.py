
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "Linear SVM": "linear_svm.joblib",
    "Isolation Forest": "isolation_forest.joblib",
}


def get_scores(model_name, model, X):
    """Return scores where larger values mean stronger fraud evidence."""

    if model_name == "Isolation Forest":
        return -model.decision_function(X)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    raise TypeError(
        f"Unsupported scoring interface for {model_name}"
    )


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    validation_path = DATA_DIR / "validation.csv"

    if not validation_path.exists():
        raise FileNotFoundError(validation_path)

    validation = pd.read_csv(validation_path)
    X_val = validation.drop(columns=[TARGET])
    y_val = validation[TARGET].astype(int)

    results = []

    print("\n========== VALIDATION THRESHOLD TUNING ==========")
    print("Objective: maximize validation F1-score")
    print("The test set will not be used.")

    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            print(f"Missing model: {model_path}")
            continue

        model = joblib.load(model_path)
        scores = get_scores(model_name, model, X_val)

        if not np.isfinite(scores).all():
            raise ValueError(
                f"Non-finite scores found for {model_name}"
            )

        # Every score is oriented so higher means more likely fraud.
        precision_values, recall_values, thresholds = (
            precision_recall_curve(y_val, scores)
        )

        if len(thresholds) == 0:
            raise ValueError(
                f"No thresholds available for {model_name}"
            )

        f1_values = (
            2 * precision_values[:-1] * recall_values[:-1]
            / (
                precision_values[:-1] + recall_values[:-1]
                + 1e-12
            )
        )

        best_f1 = f1_values.max()
        best_indices = np.flatnonzero(
            np.isclose(f1_values, best_f1)
        )

        # If F1 ties, choose the threshold with higher precision.
        best_index = best_indices[
            np.argmax(precision_values[best_indices])
        ]

        threshold = float(thresholds[best_index])
        predictions = (scores >= threshold).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )
        recall = recall_score(
            y_val, predictions, zero_division=0
        )
        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "model": model_name,
            "threshold": threshold,
            "average_precision": average_precision_score(
                y_val, scores
            ),
            "roc_auc": roc_auc_score(y_val, scores),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        })

        print(f"\n--- {model_name} ---")
        print(f"Selected threshold: {threshold:.6f}")
        print(f"Average precision: {average_precision_score(y_val, scores):.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"True positives: {tp}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")

    if not results:
        raise RuntimeError("No models were evaluated.")

    comparison = pd.DataFrame(results).sort_values(
        by=["f1_score", "precision"],
        ascending=[False, False],
    )

    output_path = REPORT_DIR / "threshold_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print("\n========== TUNED VALIDATION LEADERBOARD ==========")
    print(
        comparison[
            [
                "model",
                "threshold",
                "average_precision",
                "precision",
                "recall",
                "f1_score",
                "false_positives",
                "false_negatives",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {output_path}")
    print("\nThreshold tuning completed.")


if __name__ == "__main__":
    main()
