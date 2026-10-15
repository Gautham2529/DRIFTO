"""
src/simulate.py
---------------
Simulates new production data by applying controlled Gaussian noise and
mean shifts to a subset of features in the reference dataset.

Saved artefacts
---------------
  outputs/new_data.csv       — shifted feature matrix (no labels)
  outputs/new_predictions.csv — model predictions on the new data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from src.utils import (
    MODELS_DIR,
    OUTPUTS_DIR,
    ensure_dirs,
    section,
)

# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 0

# Features to drift and the strength of each drift
DRIFT_CONFIG: dict[str, dict] = {
    "mean radius": {"shift": 2.5, "noise_scale": 0.8},
    "mean texture": {"shift": 1.8, "noise_scale": 0.6},
    "mean perimeter": {"shift": 5.0, "noise_scale": 1.2},
    "mean area": {"shift": 40.0, "noise_scale": 10.0},
    "mean smoothness": {"shift": 0.01, "noise_scale": 0.003},
    "worst radius": {"shift": 3.0, "noise_scale": 1.0},
    "worst texture": {"shift": 2.0, "noise_scale": 0.7},
}


def simulate_new_data(
    reference_path: str | None = None,
    n_samples: int | None = None,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Load reference data, apply feature drift, and return the new DataFrame.

    Parameters
    ----------
    reference_path : str or None
        Path to outputs/reference_data.csv.  Defaults to the standard location.
    n_samples : int or None
        How many samples to generate.  Defaults to the size of reference data.
    random_seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        New data with shifted features.  No 'target' column.
    """
    if reference_path is None:
        reference_path = OUTPUTS_DIR / "reference_data.csv"

    ref = pd.read_csv(reference_path)
    # Drop the target column if present
    if "target" in ref.columns:
        ref = ref.drop(columns=["target"])

    rng = np.random.default_rng(random_seed)

    if n_samples is None:
        n_samples = len(ref)

    # Sample rows with replacement to create the base population
    new_df = ref.sample(n=n_samples, replace=True, random_state=random_seed).reset_index(
        drop=True
    )

    # Apply drift to configured features
    for feature, cfg in DRIFT_CONFIG.items():
        if feature not in new_df.columns:
            continue
        shift = cfg.get("shift", 0.0)
        noise_scale = cfg.get("noise_scale", 0.0)
        noise = rng.normal(loc=shift, scale=noise_scale, size=n_samples)
        new_df[feature] = new_df[feature] + noise

    return new_df


def simulate() -> None:
    ensure_dirs()

    section("Step 1 - Loading reference data & trained model")
    ref_path = OUTPUTS_DIR / "reference_data.csv"
    model_path = MODELS_DIR / "model.joblib"

    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference data not found at {ref_path}.\n"
            "Run  python -m src.train  first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Run  python -m src.train  first."
        )

    model = joblib.load(model_path)
    print(f"  Model loaded from models/model.joblib")

    section("Step 2 - Simulating new production data")
    new_data = simulate_new_data()
    print(f"  Samples generated : {len(new_data)}")
    print(f"  Drifted features  : {list(DRIFT_CONFIG.keys())}")

    new_data.to_csv(OUTPUTS_DIR / "new_data.csv", index=False)
    print(f"  [saved] outputs/new_data.csv")

    section("Step 3 - Generating predictions on new data")
    predictions = model.predict(new_data)
    pred_df = pd.DataFrame({"prediction": predictions})
    pred_df.to_csv(OUTPUTS_DIR / "new_predictions.csv", index=False)
    print(f"  [saved] outputs/new_predictions.csv")

    section("Prediction distribution (new data)")
    dist = pred_df["prediction"].value_counts(normalize=True).sort_index()
    labels = {0: "malignant", 1: "benign"}
    for cls, pct in dist.items():
        print(f"  Class {cls} ({labels.get(cls, '?')}) : {pct:.2%}")


if __name__ == "__main__":
    simulate()
