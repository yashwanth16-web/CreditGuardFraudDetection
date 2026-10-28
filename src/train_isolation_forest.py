
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
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

    # Isolation Forest is fitted using features only.
    # The Class column is never passed to fit().
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )

    print("Training Isolation Forest...")
    print(f"Training transactions: {len(X_train):,}")
    print("Fraud labels are not used to fit the model.")

    model.fit(X_train)

    # Lower decision_function scores indicate more anomalous records.
    # Negate the scores so higher values indicate greater anomaly.
    anomaly_scores = -model.decision_function(X_val)

    # Use the model's own default anomaly classification.
    # IsolationForest.predict returns -1 for anomalies and 1 otherwise.
    raw_predictions = model.predict(X_val)
    predictions = (raw_predictions == -1).astype(int)

    average_precision = average_precision_score(
        y_val, anomaly_scores
    )
    roc_auc = roc_auc_score(y_val, anomaly_scores)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(y_val, predictions, zero_division=0)

    print("\n========== ISOLATION FOREST RESULTS ==========")
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

    model_path = MODEL_DIR / "isolation_forest.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "model": "Isolation Forest",
        "threshold_method": "IsolationForest default",
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predicted_anomalies": int(predictions.sum()),
    }

    metrics_path = REPORT_DIR / "isolation_forest_validation.csv"
    pd.DataFrame([metrics]).to_csv(
        metrics_path,
        index=False,
    )

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("\nIsolation Forest training completed!")


if __name__ == "__main__":
    main()
