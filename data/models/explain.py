"""
SHAP-based model explanations for F1 predictions.

Provides:
1. Global feature importance (SHAP-based, more reliable than built-in)
2. Per-prediction explanations ("Why did model predict VER P1?")
3. Feature interaction analysis

Usage:
    python -m data.models.explain                          # Global importance
    python -m data.models.explain --driver max_verstappen  # Driver-specific
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "cache" / "models"


def _get_shap_values(
    model, X_explain: pd.DataFrame, X_background: pd.DataFrame
) -> np.ndarray:
    """Compute SHAP values with TreeExplainer, falling back to KernelExplainer."""
    import shap

    try:
        return shap.TreeExplainer(model).shap_values(X_explain)
    except (TypeError, ValueError, AttributeError):
        n = min(100, len(X_background))
        logger.warning("TreeExplainer failed, using KernelExplainer (slower)")
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_background, n))
        return explainer.shap_values(X_explain, nsamples=n)


def compute_shap_importance(
    model,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> pd.DataFrame:
    """
    Compute SHAP-based global feature importance.

    More reliable than built-in feature_importances_ because SHAP
    accounts for feature interactions and correlations.

    Args:
        model: Trained model (position_model from F1Predictor)
        X: Feature matrix
        max_samples: Max samples for SHAP computation (speed vs accuracy)

    Returns:
        DataFrame with feature, shap_importance, direction columns
    """
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    shap_values = _get_shap_values(model, X_sample, X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame({
        "feature": X.columns,
        "shap_importance": mean_abs_shap,
    }).sort_values("shap_importance", ascending=False)

    mean_shap = shap_values.mean(axis=0)
    importance["direction"] = ["worse" if v > 0 else "better" for v in mean_shap]

    return importance


def explain_prediction(
    model,
    X: pd.DataFrame,
    driver_idx: int,
    feature_names: list,
    top_n: int = 10,
    shap_values_all: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Explain a single driver's prediction.

    Returns top features driving this prediction with their SHAP contributions.
    If shap_values_all is provided, uses pre-computed values (avoids re-creating explainer).
    """
    if shap_values_all is not None:
        sv = shap_values_all[driver_idx]
    else:
        sv = _get_shap_values(model, X.iloc[[driver_idx]], X)[0]

    contributions = pd.DataFrame({
        "feature": feature_names,
        "value": X.iloc[driver_idx].values,
        "shap_contribution": sv,
    })

    contributions["abs_contribution"] = contributions["shap_contribution"].abs()
    contributions = contributions.sort_values("abs_contribution", ascending=False)

    return contributions.head(top_n)


def explain_race(
    predictor,
    X_race: pd.DataFrame,
    driver_ids: list,
    top_n: int = 5,
) -> dict:
    """
    Explain predictions for all drivers in a race.

    Builds SHAP explainer once and computes values for all drivers in a single call.
    Returns dict of driver_id -> top contributing features.
    """
    features = X_race[predictor.feature_names]

    # Compute SHAP values for all drivers at once (single explainer creation)
    shap_values = _get_shap_values(predictor.position_model, features, features)

    results = {}
    for i, driver_id in enumerate(driver_ids):
        try:
            explanation = explain_prediction(
                predictor.position_model, features, i,
                predictor.feature_names, top_n=top_n,
                shap_values_all=shap_values,
            )
            results[driver_id] = explanation
        except Exception as e:
            logger.warning("Failed to explain %s: %s", driver_id, e)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explain F1 predictions")
    parser.add_argument("--driver", type=str, default=None)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from data.models.predictor import F1Predictor
    from data.features.engineer import prepare_training_data

    predictor = F1Predictor()
    predictor.load()

    fm = pd.read_parquet("data/cache/processed/feature_matrix.parquet")
    X, y = prepare_training_data(fm, target="position")

    print("Computing SHAP feature importance (this may take a minute)...")
    importance = compute_shap_importance(predictor.position_model, X)

    print(f"\nTop {args.top} features by SHAP importance:")
    print(importance.head(args.top).to_string(index=False))

    # Save
    importance.to_csv(MODEL_DIR / "shap_importance.csv", index=False)
    print(f"\nSaved to {MODEL_DIR / 'shap_importance.csv'}")
