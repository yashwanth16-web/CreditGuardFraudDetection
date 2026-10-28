
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

TARGET = "Class"
TIME_COLUMN = "Time"


def load_splits():
    """Load the saved chronological data partitions."""
    splits = {}

    for name in ["train", "validation", "test"]:
        path = DATA_DIR / f"{name}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run src/prepare_data.py first."
            )

        splits[name] = pd.read_csv(path)

    return splits


def split_features_target(df):
    """Separate model inputs from the target label."""
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].astype(int).copy()

    if X.isna().any().any():
        raise ValueError("Feature data contains missing values.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("All features must be numeric.")

    return X, y


def build_scaled_pipeline(model):
    """
    Scale features inside a pipeline.

    The scaler learns its parameters only when fit() is called
    on training data.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_unscaled_pipeline(model):
    """
    Use original feature scales for tree-based models.
    """
    return Pipeline(
        steps=[
            ("model", model),
        ]
    )


def main():
    splits = load_splits()

    print("\n========== PREPROCESSING CHECK ==========")

    for name, df in splits.items():
        X, y = split_features_target(df)

        print(f"\n{name.upper()}")
        print(f"Feature rows: {X.shape[0]:,}")
        print(f"Feature columns: {X.shape[1]}")
        print(f"Fraud cases: {int(y.sum()):,}")
        print(f"Legitimate cases: {int((y == 0).sum()):,}")

    print("\nPreprocessing module is ready.")
    print("No scaler has been fitted.")
    print("No original or split data has been modified.")


if __name__ == "__main__":
    main()
