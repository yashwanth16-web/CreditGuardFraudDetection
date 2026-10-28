
from pathlib import Path

import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "Class"
THRESHOLD = 0.50


def load_data(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run src/prepare_data.py first."
        )

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
        raise ValueError(
            "Training and validation features do not match."
        )

    # Decision Trees generally do not require feature scaling.
    model = Pipeline(
        steps=[
            (
                "classifier",
                DecisionTreeClassifier(
                    class_weight="balanced",
                    max_depth=8,
                    min_samples_leaf=10,
                    random_state=42,
                ),
            )
        ]
    )

    print("Training Decision Tree...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    average_precision = average_precision_score(
        y_val, probabilities
    )
    roc_auc = roc_auc_score(y_val, probabilities)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    print("\n========== DECISION TREE RESULTS ==========")
    print(f"Threshold: {THRESHOLD:.2f}")
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

    model_path = MODEL_DIR / "decision_tree.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "model": "Decision Tree",
        "threshold": THRESHOLD,
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    metrics_path = REPORT_DIR / "decision_tree_validation.csv"
    pd.DataFrame([metrics]).to_csv(
        metrics_path,
        index=False,
    )

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("\nDecision Tree training completed!")


if __name__ == "__main__":
    main()
