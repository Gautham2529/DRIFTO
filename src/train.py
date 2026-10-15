"""
src/train.py
------------
Trains a RandomForestClassifier on the Breast Cancer dataset,
evaluates baseline performance, and saves artefacts to disk.

Saved artefacts
---------------
  models/model.joblib          — trained pipeline
  outputs/baseline_metrics.json — accuracy, precision, recall, F1
  outputs/reference_data.csv   — test (reference) split used downstream
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_dataset, split_data
from src.utils import (
    MODELS_DIR,
    OUTPUTS_DIR,
    ensure_dirs,
    save_json,
    section,
)


def build_pipeline() -> Pipeline:
    """Return an sklearn Pipeline with scaling + RandomForest classifier."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute classification metrics and return them as a dict."""
    y_pred = model.predict(X)
    return {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision_weighted": round(
            precision_score(y, y_pred, average="weighted", zero_division=0), 4
        ),
        "recall_weighted": round(
            recall_score(y, y_pred, average="weighted", zero_division=0), 4
        ),
        "f1_weighted": round(
            f1_score(y, y_pred, average="weighted", zero_division=0), 4
        ),
    }


def train() -> None:
    ensure_dirs()

    section("Step 1 - Loading dataset")
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")

    section("Step 2 - Training RandomForestClassifier")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("  Training complete.")

    section("Step 3 - Evaluating on test (reference) split")
    metrics = evaluate(pipeline, X_test, y_test)
    for k, v in metrics.items():
        print(f"  {k:<26} {v}")

    section("Step 4 - Saving artefacts")
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"  [saved] models/model.joblib")

    save_json(metrics, OUTPUTS_DIR / "baseline_metrics.json")

    # Save reference (test) split for drift comparison later
    ref = X_test.copy()
    ref["target"] = y_test.values
    ref.to_csv(OUTPUTS_DIR / "reference_data.csv", index=False)
    print(f"  [saved] outputs/reference_data.csv")

    section("Done - baseline metrics")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1       : {metrics['f1_weighted']:.4f}")


if __name__ == "__main__":
    train()
