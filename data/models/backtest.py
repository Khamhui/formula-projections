"""
Walk-forward backtesting for F1 predictions.

Instead of a single train/test split, this evaluates the model on each race
sequentially: for race N, train on all races before N, predict N, record accuracy.

This reveals:
- True out-of-sample performance per race
- Where the model fails (specific circuits, conditions)
- Consistency of predictions across the season

Usage:
    python -m data.models.backtest                     # Backtest 2025
    python -m data.models.backtest --season 2024       # Specific season
    python -m data.models.backtest --season 2024 2025  # Multiple seasons
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "cache" / "processed"


def walk_forward_backtest(
    feature_matrix: pd.DataFrame,
    test_seasons: list,
    min_train_races: int = 100,
) -> pd.DataFrame:
    """
    Walk-forward evaluation: for each race in test_seasons,
    train on all prior data and predict that race.

    Args:
        feature_matrix: Full feature matrix from build_feature_matrix()
        test_seasons: Seasons to evaluate (e.g., [2024, 2025])
        min_train_races: Minimum training races before starting predictions

    Returns:
        DataFrame with per-race metrics and predictions
    """
    from data.features.engineer import prepare_training_data
    from data.models.predictor import _create_model

    results = []

    # Get all unique races in chronological order
    races = (
        feature_matrix[["season", "round", "circuit_id"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
    )

    # Only evaluate races in test_seasons
    test_races = races[races["season"].isin(test_seasons)]
    all_race_keys = list(zip(races["season"], races["round"]))

    for idx, (_, race) in enumerate(test_races.iterrows()):
        season = int(race["season"])
        rnd = int(race["round"])
        circuit_id = race["circuit_id"]

        # Training data: all races BEFORE this one
        train_mask = (
            (feature_matrix["season"] < season) |
            ((feature_matrix["season"] == season) & (feature_matrix["round"] < rnd))
        )
        test_mask = (
            (feature_matrix["season"] == season) &
            (feature_matrix["round"] == rnd)
        )

        train_data = feature_matrix[train_mask]
        test_data = feature_matrix[test_mask]

        if len(train_data) < min_train_races * 15:  # ~15 drivers per race
            continue

        try:
            X_train, y_train = prepare_training_data(train_data, target="position")
            X_test, y_test = prepare_training_data(test_data, target="position")
        except Exception as e:
            logger.warning("Skipping S%d R%02d: %s", season, rnd, e)
            continue

        if X_test.empty or y_test.empty:
            continue

        # Align columns
        for col in set(X_train.columns) - set(X_test.columns):
            X_test[col] = 0
        for col in set(X_test.columns) - set(X_train.columns):
            X_train[col] = 0
        X_test = X_test[X_train.columns]

        # Train a quick model (no ensemble for speed)
        model = _create_model(
            "regressor", n_estimators=500, max_depth=6,
            learning_rate=0.05, random_state=42,
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)

        # Position ranking
        pred_ranks = pd.Series(y_pred, index=y_test.index).rank()
        actual_ranks = y_test.rank()

        # Spearman rank correlation
        spearman_corr, _ = spearmanr(actual_ranks, pred_ranks)

        # Correct winner? (compare by index, not positional offset)
        pred_series = pd.Series(y_pred, index=y_test.index)
        correct_winner = int(pred_series.idxmin() == y_test.idxmin())

        # Correct podium (top 3 by index)
        pred_top3 = set(pred_series.nsmallest(3).index)
        actual_top3 = set(y_test.nsmallest(3).index)
        podium_overlap = len(pred_top3 & actual_top3) / 3

        results.append({
            "season": season,
            "round": rnd,
            "circuit_id": circuit_id,
            "n_drivers": len(y_test),
            "n_train_races": len(train_data) // 15,  # approximate
            "mae": mae,
            "spearman_corr": spearman_corr,
            "correct_winner": correct_winner,
            "podium_overlap": podium_overlap,
        })

        logger.info(
            "S%d R%02d %-20s MAE=%.2f Spearman=%.3f Winner=%s Podium=%.0f%%",
            season, rnd, circuit_id[:20], mae, spearman_corr,
            "✓" if correct_winner else "✗", podium_overlap * 100,
        )

    df = pd.DataFrame(results)

    if not df.empty:
        logger.info("\n=== Walk-Forward Backtest Summary ===")
        logger.info("Races evaluated: %d", len(df))
        logger.info("Mean MAE: %.3f (±%.3f)", df["mae"].mean(), df["mae"].std())
        logger.info("Mean Spearman: %.3f", df["spearman_corr"].mean())
        logger.info("Winner correct: %d/%d (%.1f%%)",
                    df["correct_winner"].sum(), len(df),
                    df["correct_winner"].mean() * 100)
        logger.info("Mean podium overlap: %.1f%%", df["podium_overlap"].mean() * 100)

        # Per-season breakdown
        for season, sdf in df.groupby("season"):
            logger.info(
                "  %d: MAE=%.3f, Winner=%.0f%%, Podium=%.0f%%, Spearman=%.3f",
                season, sdf["mae"].mean(),
                sdf["correct_winner"].mean() * 100,
                sdf["podium_overlap"].mean() * 100,
                sdf["spearman_corr"].mean(),
            )

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--season", type=int, nargs="+", default=[2025])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    fm = pd.read_parquet(DATA_DIR / "feature_matrix.parquet")

    results = walk_forward_backtest(fm, test_seasons=args.season)

    if not results.empty:
        results.to_csv(DATA_DIR / "backtest_results.csv", index=False)
        print(f"\nResults saved to {DATA_DIR / 'backtest_results.csv'}")
