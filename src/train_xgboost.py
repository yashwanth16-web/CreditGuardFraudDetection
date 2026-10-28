
from pathlib import Path

import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "Class"
THRESHOLD = 0.50


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

    # Give the rare fraud class greater weight during training.
    fraud_count = int((y_train == 1).sum())
    legitimate_count = int((y_train == 0).sum())
    scale_pos_weight = legitimate_count / fraud_count

    model = Pipeline(
        steps=[
            (
                "classifier",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=2,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=42,
                ),
            )
        ]
    )

    print("Training XGBoost...")
    print(f"Training fraud cases: {fraud_count}")
    print(f"Training legitimate cases: {legitimate_count}")
    print(f"Scale positive weight: {scale_pos_weight:.2f}")

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
    f1 = f1_score(y_val, predictions, zero_division=0)

    print("\n========== XGBOOST RESULTS ==========")
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

    model_path = MODEL_DIR / "xgboost.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "model": "XGBoost",
        "threshold": THRESHOLD,
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    metrics_path = REPORT_DIR / "xgboost_validation.csv"
    pd.DataFrame([metrics]).to_csv(
        metrics_path,
        index=False,
    )

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("\nXGBoost training completed!")


if __name__ == "__main__":
    main()
