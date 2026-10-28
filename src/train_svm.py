
from pathlib import Path

import joblib
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


def load_data(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    return pd.read_csv(path)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_data("train.csv")
    validation = load_data("validation.csv")

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET].astype(int)

    X_val = validation.drop(columns=[TARGET])
    y_val = validation[TARGET].astype(int)

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Training and validation columns differ.")

    # LinearSVC is suitable for a large training set.
    # StandardScaler is fitted only on the training data.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    C=1.0,
                    max_iter=5000,
                    dual="auto",
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training Linear SVM...")
    model.fit(X_train, y_train)

    # LinearSVC returns decision scores, not probabilities.
    scores = model.decision_function(X_val)

    # LinearSVC's default decision boundary is score = 0.
    predictions = (scores >= 0.0).astype(int)

    average_precision = average_precision_score(y_val, scores)
    roc_auc = roc_auc_score(y_val, scores)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(y_val, predictions, zero_division=0)

    print("\n========== LINEAR SVM RESULTS ==========")
    print("Decision threshold: 0.00 (decision score)")
    print(f"Average precision: {average_precision:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    model_path = MODEL_DIR / "linear_svm.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "model": "Linear SVM",
        "decision_threshold": 0.0,
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    metrics_path = REPORT_DIR / "linear_svm_validation.csv"
    pd.DataFrame([metrics]).to_csv(
        metrics_path,
        index=False,
    )

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("\nLinear SVM training completed!")


if __name__ == "__main__":
    main()
