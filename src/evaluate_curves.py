
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"

TARGET = "Class"

MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "Linear SVM": "linear_svm.joblib",
    "Isolation Forest": "isolation_forest.joblib",
}


def get_scores(name, model, X):
    if name == "Isolation Forest":
        return -model.decision_function(X)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    raise ValueError(f"Unsupported model: {name}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(DATA_DIR / "validation.csv")
    X_val = validation.drop(columns=[TARGET])
    y_val = validation[TARGET].astype(int)

    fig, ax = plt.subplots(figsize=(10, 7))
    results = []

    for name, filename in MODELS.items():
        path = MODEL_DIR / filename

        if not path.exists():
            print(f"Skipping missing model: {path}")
            continue

        model = joblib.load(path)
        scores = get_scores(name, model, X_val)

        average_precision = average_precision_score(y_val, scores)

        PrecisionRecallDisplay.from_predictions(
            y_val,
            scores,
            name=name,
            ax=ax,
        )

        results.append({
            "model": name,
            "average_precision": average_precision,
        })

    ax.set_title("Precision–Recall Curves — Validation Set")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output = FIGURE_DIR / "precision_recall_curves.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    results_path = ROOT / "reports" / "precision_recall_metrics.csv"
    pd.DataFrame(results).sort_values(
        "average_precision", ascending=False
    ).to_csv(results_path, index=False)

    print(f"Saved chart: {output}")
    print(f"Saved metrics: {results_path}")
    print(pd.DataFrame(results).sort_values(
        "average_precision", ascending=False
    ).to_string(index=False))


if __name__ == "__main__":
    main()
