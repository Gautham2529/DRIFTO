"""
src/monitor.py
--------------
Computes feature-level drift between the reference data and the new
simulated production data, flags drifted features, and saves a
comprehensive monitoring report.

Saved artefacts
---------------
  outputs/drift_report.json  — per-feature drift scores + flagged list
  outputs/monitoring_summary.json — high-level monitoring summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import (
    OUTPUTS_DIR,
    ensure_dirs,
    load_json,
    save_json,
    section,
)

# Features whose drift score (|Δmean| / ref_std) exceeds this threshold
# are flagged as drifted.
DRIFT_THRESHOLD = 0.5


def compute_drift_scores(
    reference: pd.DataFrame,
    new_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-feature drift statistics.

    Drift score  =  |mean_new − mean_ref|  /  std_ref

    Parameters
    ----------
    reference : pd.DataFrame
        Reference (test) feature matrix.
    new_data : pd.DataFrame
        Simulated production feature matrix.

    Returns
    -------
    pd.DataFrame with columns:
        feature, ref_mean, new_mean, ref_std, new_std,
        mean_diff, drift_score, drifted
    """
    # Only compare common numeric columns
    common_cols = [
        c for c in reference.columns
        if c in new_data.columns and c != "target"
        and pd.api.types.is_numeric_dtype(reference[c])
    ]

    records = []
    for col in common_cols:
        ref_mean = reference[col].mean()
        new_mean = new_data[col].mean()
        ref_std = reference[col].std()
        new_std = new_data[col].std()
        mean_diff = abs(new_mean - ref_mean)
        drift_score = mean_diff / ref_std if ref_std > 0 else 0.0

        records.append(
            {
                "feature": col,
                "ref_mean": round(ref_mean, 4),
                "new_mean": round(new_mean, 4),
                "ref_std": round(ref_std, 4),
                "new_std": round(new_std, 4),
                "mean_diff": round(mean_diff, 4),
                "drift_score": round(drift_score, 4),
                "drifted": drift_score >= DRIFT_THRESHOLD,
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values("drift_score", ascending=False).reset_index(drop=True)
    return df


def prediction_distribution(predictions: pd.Series) -> dict[str, float]:
    """Return class probability dict from a prediction Series."""
    dist = predictions.value_counts(normalize=True).sort_index()
    return {str(int(k)): round(float(v), 4) for k, v in dist.items()}


def monitor() -> None:
    ensure_dirs()

    section("Step 1 - Loading artefacts")
    ref_path = OUTPUTS_DIR / "reference_data.csv"
    new_data_path = OUTPUTS_DIR / "new_data.csv"
    new_pred_path = OUTPUTS_DIR / "new_predictions.csv"
    baseline_path = OUTPUTS_DIR / "baseline_metrics.json"

    for p in [ref_path, new_data_path, new_pred_path, baseline_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run  python -m src.train  then  python -m src.simulate  first."
            )

    reference = pd.read_csv(ref_path)
    if "target" in reference.columns:
        reference = reference.drop(columns=["target"])

    new_data = pd.read_csv(new_data_path)
    new_preds = pd.read_csv(new_pred_path)["prediction"]
    baseline_metrics = load_json(baseline_path)

    print(f"  Reference samples : {len(reference)}")
    print(f"  New data  samples : {len(new_data)}")

    section("Step 2 - Computing feature drift scores")
    drift_df = compute_drift_scores(reference, new_data)

    drifted_features = drift_df[drift_df["drifted"]]["feature"].tolist()
    total_features = len(drift_df)
    drifted_count = len(drifted_features)

    print(f"  Total features    : {total_features}")
    print(f"  Drifted features  : {drifted_count}  (threshold >= {DRIFT_THRESHOLD})")

    section("Step 3 - Drift Report (top 10)")
    print(
        drift_df.head(10).to_string(
            index=False,
            columns=["feature", "ref_mean", "new_mean", "drift_score", "drifted"],
        )
    )

    section("Step 4 - Prediction distribution on new data")
    pred_dist = prediction_distribution(new_preds)
    class_labels = {"0": "malignant", "1": "benign"}
    for cls, pct in pred_dist.items():
        print(f"  Class {cls} ({class_labels.get(cls, '?')}) : {pct:.2%}")

    section("Step 5 - Saving monitoring outputs")

    # Drift report JSON
    drift_report = {
        "drift_threshold": DRIFT_THRESHOLD,
        "total_features": total_features,
        "drifted_feature_count": drifted_count,
        "drifted_features": drifted_features,
        "feature_stats": drift_df.to_dict(orient="records"),
    }
    save_json(drift_report, OUTPUTS_DIR / "drift_report.json")

    # High-level monitoring summary
    monitoring_summary = {
        "baseline_metrics": baseline_metrics,
        "new_data_prediction_distribution": pred_dist,
        "drift_summary": {
            "threshold": DRIFT_THRESHOLD,
            "total_features": total_features,
            "drifted_count": drifted_count,
            "drifted_features": drifted_features,
        },
    }
    save_json(monitoring_summary, OUTPUTS_DIR / "monitoring_summary.json")

    section("Monitoring Complete")
    print(f"  Baseline accuracy : {baseline_metrics.get('accuracy', 'N/A')}")
    print(f"  Baseline F1       : {baseline_metrics.get('f1_weighted', 'N/A')}")
    print(f"  Drifted features  : {drifted_features}")


if __name__ == "__main__":
    monitor()
