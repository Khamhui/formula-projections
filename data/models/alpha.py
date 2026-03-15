"""
Market Alpha Analysis — measures model's edge over bookmaker odds.

Compares model win probabilities against market consensus to quantify
predictive alpha. Tracks cumulative ROI from a simulated Kelly betting
strategy across backtested races.

Usage:
    python -m data.models.alpha --season 2024 2025
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data.models.value import ValueDetector, brier_score

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache" / "processed"


class AlphaTracker:
    """Tracks cumulative model alpha vs market across multiple races."""

    def __init__(self, kelly_fraction: float = 0.25, min_edge: float = 0.03):
        self.kelly_fraction = kelly_fraction
        self.detector = ValueDetector(min_edge=min_edge, min_prob=0.02)
        self.race_results: List[Dict] = []

    def evaluate_race(
        self,
        model_probs: pd.DataFrame,
        market_probs: pd.DataFrame,
        actual_winner: str,
        race_id: str,
    ) -> Dict:
        """
        Evaluate model vs market for a single race.

        Args:
            model_probs: DataFrame with columns: driver_id, model_win_pct (0-1 proportion)
            market_probs: DataFrame with columns: driver_id, market_win_pct (0-1 proportion)
            actual_winner: driver_id of the actual race winner
            race_id: Unique race identifier (e.g. "2024_R03")

        Returns:
            Dict with race-level alpha metrics
        """
        merged = pd.merge(model_probs, market_probs, on="driver_id", how="inner")
        if merged.empty:
            return {"race_id": race_id, "n_drivers": 0, "alpha_brier": np.nan}

        # Build outcome vector
        outcomes = (merged["driver_id"] == actual_winner).astype(float).values
        model_p = merged["model_win_pct"].values
        market_p = merged["market_win_pct"].values

        # Brier scores
        brier_model = brier_score(model_p, outcomes)
        brier_market = brier_score(market_p, outcomes)
        alpha_brier = brier_market - brier_model  # positive = model better

        # Log loss
        eps = 1e-15
        log_loss_model = -np.mean(outcomes * np.log(np.clip(model_p, eps, 1)) + (1 - outcomes) * np.log(np.clip(1 - model_p, eps, 1)))
        log_loss_market = -np.mean(outcomes * np.log(np.clip(market_p, eps, 1)) + (1 - outcomes) * np.log(np.clip(1 - market_p, eps, 1)))

        # Value bets and Kelly stakes
        value_bets = self.detector.find_value(
            model_probs[["driver_id", "model_win_pct"]],
            market_probs[["driver_id", "market_win_pct"]],
        )

        # Simulate Kelly P&L
        race_staked = 0.0
        race_return = 0.0
        n_bets = 0
        for _, bet in value_bets.iterrows():
            kelly = self.detector.kelly_fraction(
                bet["model_win_pct"], bet["market_win_pct"], fraction=self.kelly_fraction,
            )
            if kelly > 0:
                n_bets += 1
                race_staked += kelly
                if bet["driver_id"] == actual_winner:
                    decimal_odds = 1.0 / bet["market_win_pct"] if bet["market_win_pct"] > 0 else 0
                    race_return += kelly * decimal_odds

        race_pnl = race_return - race_staked

        # Winner prediction accuracy
        model_predicted_winner = merged.loc[merged["model_win_pct"].idxmax(), "driver_id"]
        market_predicted_winner = merged.loc[merged["market_win_pct"].idxmax(), "driver_id"]

        result = {
            "race_id": race_id,
            "n_drivers": len(merged),
            "brier_model": brier_model,
            "brier_market": brier_market,
            "alpha_brier": alpha_brier,
            "log_loss_model": log_loss_model,
            "log_loss_market": log_loss_market,
            "alpha_log_loss": log_loss_market - log_loss_model,
            "n_value_bets": n_bets,
            "race_staked": race_staked,
            "race_return": race_return,
            "race_pnl": race_pnl,
            "model_correct_winner": model_predicted_winner == actual_winner,
            "market_correct_winner": market_predicted_winner == actual_winner,
        }

        self.race_results.append(result)
        return result

    def cumulative_report(self) -> Dict:
        """Generate cumulative alpha report across all evaluated races."""
        if not self.race_results:
            return {}

        df = pd.DataFrame(self.race_results)
        valid = df.dropna(subset=["alpha_brier"])

        total_staked = df["race_staked"].sum()
        total_return = df["race_return"].sum()
        total_pnl = total_return - total_staked

        return {
            "n_races": len(valid),
            "mean_brier_model": valid["brier_model"].mean(),
            "mean_brier_market": valid["brier_market"].mean(),
            "mean_alpha_brier": valid["alpha_brier"].mean(),
            "mean_log_loss_model": valid["log_loss_model"].mean(),
            "mean_log_loss_market": valid["log_loss_market"].mean(),
            "mean_alpha_log_loss": valid["alpha_log_loss"].mean(),
            "model_beats_market_pct": (valid["alpha_brier"] > 0).mean() * 100,
            "model_winner_accuracy": df["model_correct_winner"].mean() * 100,
            "market_winner_accuracy": df["market_correct_winner"].mean() * 100,
            "total_value_bets": int(df["n_value_bets"].sum()),
            "total_staked": total_staked,
            "total_return": total_return,
            "total_pnl": total_pnl,
            "roi": total_pnl / total_staked if total_staked > 0 else 0.0,
        }

    def per_race_dataframe(self) -> pd.DataFrame:
        """Return per-race results as a DataFrame."""
        return pd.DataFrame(self.race_results)


def backtest_alpha(
    backtest_predictions: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    kelly_fraction: float = 0.25,
) -> Dict:
    """
    Run alpha analysis on walk-forward backtest results.

    Loads cached odds for each backtested race and compares model
    probabilities against market consensus.

    Args:
        backtest_predictions: Driver-level backtest predictions (from backtest.py attrs)
            Must have: season, round, driver_id, prob_win, actual_position
        feature_matrix: Full feature matrix (used to resolve driver_id mapping)
        kelly_fraction: Kelly fraction for stake sizing

    Returns:
        Dict with cumulative alpha metrics + per-race DataFrame
    """
    from data.ingest.odds import OddsClient

    client = OddsClient()
    tracker = AlphaTracker(kelly_fraction=kelly_fraction)

    races = backtest_predictions.groupby(["season", "round"])

    n_with_odds = 0
    n_without_odds = 0

    for (season, rnd), group in races:
        race_id = f"{int(season)}_R{int(rnd):02d}"

        # Load cached odds
        odds = client.load_odds(int(season), int(rnd))
        if odds is None or odds.empty:
            n_without_odds += 1
            continue

        n_with_odds += 1
        consensus = OddsClient.consensus_odds(odds)

        # Model probabilities from backtest
        model_probs = group[["driver_id", "prob_win"]].copy()
        model_probs = model_probs.dropna(subset=["prob_win"])
        if model_probs.empty:
            continue

        # Normalise model probs to sum to 1
        total = model_probs["prob_win"].sum()
        if total > 0:
            model_probs["model_win_pct"] = model_probs["prob_win"] / total
        else:
            continue

        # Market probabilities
        market_probs = consensus[["driver_id", "market_win_pct"]].copy()

        # Actual winner
        winners = group[group["actual_position"] == 1.0]
        if winners.empty:
            continue
        actual_winner = winners.iloc[0]["driver_id"]

        tracker.evaluate_race(
            model_probs[["driver_id", "model_win_pct"]],
            market_probs,
            actual_winner,
            race_id,
        )

    report = tracker.cumulative_report()
    report["races_with_odds"] = n_with_odds
    report["races_without_odds"] = n_without_odds

    logger.info("Alpha backtest: %d races with odds, %d without", n_with_odds, n_without_odds)
    if report:
        logger.info("  Mean Brier (model): %.4f", report.get("mean_brier_model", 0))
        logger.info("  Mean Brier (market): %.4f", report.get("mean_brier_market", 0))
        logger.info("  Alpha (Brier): %.4f", report.get("mean_alpha_brier", 0))
        logger.info("  Model beats market: %.1f%% of races", report.get("model_beats_market_pct", 0))
        logger.info("  Kelly ROI: %.2f%%", report.get("roi", 0) * 100)

    return {
        "summary": report,
        "per_race": tracker.per_race_dataframe(),
    }


def compute_race_alpha(
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    actual_outcomes: np.ndarray,
) -> Dict[str, float]:
    """
    Quick race-level alpha computation (no Kelly, just accuracy comparison).

    Args:
        model_probs: Array of model win probabilities per driver
        market_probs: Array of market win probabilities per driver
        actual_outcomes: Binary array (1 for winner, 0 for rest)

    Returns:
        Dict with brier_model, brier_market, alpha_brier
    """
    brier_m = brier_score(model_probs, actual_outcomes)
    brier_mkt = brier_score(market_probs, actual_outcomes)

    return {
        "brier_model": brier_m,
        "brier_market": brier_mkt,
        "alpha_brier": brier_mkt - brier_m,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Market Alpha Analysis")
    parser.add_argument("--season", type=int, nargs="+", default=[2025])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load backtest predictions
    pred_path = CACHE_DIR / "backtest_driver_predictions.csv"
    if not pred_path.exists():
        print(f"No backtest predictions found at {pred_path}")
        print("Run: python -m data.models.backtest --season", *args.season)
        exit(1)

    predictions = pd.read_csv(pred_path)
    fm = pd.read_parquet(CACHE_DIR / "feature_matrix.parquet")

    result = backtest_alpha(predictions, fm)

    if result["summary"]:
        print("\n=== Market Alpha Report ===")
        for k, v in result["summary"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    per_race = result["per_race"]
    if not per_race.empty:
        per_race.to_csv(CACHE_DIR / "alpha_per_race.csv", index=False)
        print(f"\nPer-race results saved to {CACHE_DIR / 'alpha_per_race.csv'}")
