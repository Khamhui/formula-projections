"""
Hyperparameter tuning for F1 prediction models using Optuna.

Finds optimal parameters via Bayesian optimization with TimeSeriesSplit CV.
Results are saved to JSON and reused by the predictor.

Usage:
    python -m data.models.tuner                    # Quick tune (50 trials)
    python -m data.models.tuner --n-trials 200     # Thorough tune
    python -m data.models.tuner --model position   # Tune specific model
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

logger = logging.getLogger(__name__)

PARAMS_DIR = Path(__file__).parent.parent / "cache" / "models"


def _create_model_with_params(trial: optuna.Trial, task: str = "regressor"):
    """Create a gradient boosting model with Optuna-suggested params."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    is_reg = task == "regressor"

    # Use the same model creation as predictor to ensure tuned params match
    from data.models.predictor import _create_model
    model = _create_model(
        task,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        min_child_weight=params["min_child_weight"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=42,
        n_jobs=-1,
    )
    return model, params


def tune_position_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 100,
) -> dict:
    """Tune position regression model."""
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        model, _ = _create_model_with_params(trial, "regressor")
        scores = cross_val_score(
            model, X, y, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        return -scores.mean()  # Minimize MAE

    study = optuna.create_study(direction="minimize", study_name="position_model")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Position model best MAE: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)
    return study.best_params


def tune_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
    n_trials: int = 50,
) -> dict:
    """Tune a binary classifier (podium/winner/points/DNF)."""
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        model, _ = _create_model_with_params(trial, "classifier")
        scores = cross_val_score(
            model, X, y, cv=tscv, scoring="accuracy", n_jobs=-1
        )
        return scores.mean()  # Maximize accuracy

    study = optuna.create_study(direction="maximize", study_name=f"{name}_model")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("%s model best accuracy: %.4f", name, study.best_value)
    logger.info("Best params: %s", study.best_params)
    return study.best_params


def tune_all(
    feature_matrix: pd.DataFrame,
    n_trials: int = 100,
    output_dir: Optional[Path] = None,
) -> dict[str, dict]:
    """
    Tune all model hyperparameters and save results.

    Returns dict of model_name -> best_params.
    """
    from data.features.engineer import prepare_training_data

    out = output_dir or PARAMS_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Prepare data
    X, y_position = prepare_training_data(feature_matrix, target="position")
    y_podium = (y_position <= 3).astype(int)
    y_winner = (y_position == 1).astype(int)
    y_points = (y_position <= 10).astype(int)

    # Suppress Optuna logging noise
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    all_params: dict[str, dict] = {}

    # Tune position model (most important)
    logger.info("Tuning position regression model...")
    all_params["position"] = tune_position_model(X, y_position, n_trials=n_trials)

    # Tune classifiers (fewer trials since less critical)
    classifier_trials = max(n_trials // 2, 30)
    for name, y_target in [
        ("podium", y_podium),
        ("winner", y_winner),
        ("points", y_points),
    ]:
        logger.info("Tuning %s classifier...", name)
        all_params[name] = tune_classifier(
            X, y_target, name, n_trials=classifier_trials
        )

    # Save
    params_path = out / "tuned_params.json"
    with open(params_path, "w") as f:
        json.dump(all_params, f, indent=2)
    logger.info("Tuned parameters saved to %s", params_path)

    return all_params


def load_tuned_params(path: Optional[Path] = None) -> Optional[dict[str, dict]]:
    """Load previously tuned parameters."""
    params_path = (path or PARAMS_DIR) / "tuned_params.json"
    if params_path.exists():
        with open(params_path) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune F1 prediction models")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["position", "podium", "winner", "points"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    feature_matrix = pd.read_parquet("data/cache/processed/feature_matrix.parquet")

    params = tune_all(feature_matrix, n_trials=args.n_trials)

    print("\nOptimal parameters:")
    for model_name, model_params in params.items():
        print(f"\n{model_name}:")
        for k, v in model_params.items():
            print(f"  {k}: {v}")
