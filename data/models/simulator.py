"""
Monte Carlo Race Simulator — transforms point predictions into distributions.

Instead of "Verstappen finishes P1", produces:
- P(win) = 45%, P(podium) = 78%, P(points) = 95%
- Expected points: 18.3 (±4.2)
- Full position distribution: [45%, 22%, 11%, 8%, ...]

Uses calibrated probabilities from F1Predictor for:
1. DNF sampling (independent per driver)
2. Position sampling (correlated via predicted positions + noise)
3. Safety car injection (historical rates per circuit)
4. Constraint enforcement (no two drivers same position)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# F1 points system (2010-present)
POINTS: Dict[int, int] = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1,
}

# Historical safety car probability by circuit type
SC_RATES: Dict[str, float] = {
    "street": 0.65,       # Monaco, Baku, etc — high SC rate
    "high_speed": 0.40,   # Monza, Spa
    "technical": 0.35,    # Hungary, Barcelona
    "mixed": 0.45,        # Default
}


class RaceSimulator:
    """Monte Carlo race simulator for F1 predictions."""

    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

    def simulate_race(
        self,
        predictions: pd.DataFrame,
        circuit_type: str = "mixed",
    ) -> pd.DataFrame:
        """
        Run Monte Carlo simulation on model predictions.

        Args:
            predictions: DataFrame from F1Predictor.predict_race() with columns:
                - predicted_position (float)
                - prob_podium (float)
                - prob_winner (float)
                - prob_dnf (float)
                - driver_id (str)
                Plus any other columns (passed through)
            circuit_type: For safety car probability lookup

        Returns:
            DataFrame with simulation results per driver:
                - All original columns
                - sim_win_pct: % of simulations won
                - sim_podium_pct: % on podium
                - sim_points_pct: % in points
                - sim_dnf_pct: % DNF'd
                - sim_expected_points: mean points scored
                - sim_points_std: std of points
                - sim_median_position: median finishing position
                - sim_position_25: 25th percentile position
                - sim_position_75: 75th percentile position
                - sim_position_dist: list of position probabilities [P(1st), P(2nd), ...]
        """
        n_drivers = len(predictions)
        pred_positions = predictions["predicted_position"].values.astype(float)

        if "prob_dnf" in predictions.columns:
            prob_dnf = predictions["prob_dnf"].values.astype(float)
        else:
            prob_dnf = np.zeros(n_drivers)

        sc_rate = SC_RATES.get(circuit_type, SC_RATES["mixed"])

        n_sims = self.n_simulations

        # Vectorized simulation — all 10k sims in parallel via numpy
        dnf_flags = self.rng.random((n_sims, n_drivers)) < prob_dnf
        has_sc = self.rng.random(n_sims) < sc_rate
        noise_scale = np.where(has_sc, 2.5, 1.5)[:, np.newaxis]

        performance = pred_positions + self.rng.normal(0, 1, (n_sims, n_drivers)) * noise_scale
        performance[dnf_flags] = 100 + self.rng.random(int(dnf_flags.sum())) * 10

        # Convert to positions via double-argsort (rank constraint — no ties)
        finish_positions = performance.argsort(axis=1).argsort(axis=1) + 1

        # Build per-driver statistics (vectorized)
        points_lookup = np.array([POINTS.get(p, 0) for p in range(n_drivers + 1)])
        all_points = points_lookup[finish_positions]
        all_points[dnf_flags] = 0

        sim_stats: List[Dict] = []
        for i in range(n_drivers):
            driver_positions = finish_positions[:, i]
            driver_dnfs = dnf_flags[:, i]
            classified = driver_positions[~driver_dnfs]
            n_classified = len(classified)

            # Position distribution via bincount (replaces inner loop)
            if n_classified > 0:
                counts = np.bincount(classified, minlength=n_drivers + 1)
                pos_dist = counts[1:].astype(float) / n_sims
            else:
                pos_dist = np.zeros(n_drivers)

            points_per_sim = all_points[:, i]

            sim_stats.append({
                "sim_win_pct": float((driver_positions == 1).sum()) / n_sims * 100,
                "sim_podium_pct": float((driver_positions <= 3).sum()) / n_sims * 100,
                "sim_points_pct": float(((driver_positions <= 10) & ~driver_dnfs).sum()) / n_sims * 100,
                "sim_dnf_pct": float(driver_dnfs.sum()) / n_sims * 100,
                "sim_expected_points": float(points_per_sim.mean()),
                "sim_points_std": float(points_per_sim.std()),
                "sim_median_position": float(np.median(classified)) if n_classified > 0 else float(n_drivers),
                "sim_position_25": float(np.percentile(classified, 25)) if n_classified > 0 else float(n_drivers),
                "sim_position_75": float(np.percentile(classified, 75)) if n_classified > 0 else float(n_drivers),
                "sim_position_dist": pos_dist.tolist(),
            })

        sim_df = pd.DataFrame(sim_stats, index=predictions.index)
        results = pd.concat([predictions, sim_df], axis=1)

        return results.sort_values("sim_expected_points", ascending=False)

    def simulate_championship(
        self,
        race_predictions: List[pd.DataFrame],
        circuit_types: List[str],
    ) -> pd.DataFrame:
        """
        Simulate remaining championship races.

        Args:
            race_predictions: List of prediction DataFrames (one per remaining race)
            circuit_types: Circuit type for each race

        Returns:
            Championship probability distribution per driver
        """
        if not race_predictions:
            return pd.DataFrame()

        all_drivers: set = set()
        for pred in race_predictions:
            all_drivers.update(pred["driver_id"].values)

        # Track cumulative points per simulation
        driver_total_points: Dict[str, float] = {d: 0.0 for d in all_drivers}

        for pred, ct in zip(race_predictions, circuit_types):
            race_result = self.simulate_race(pred, ct)
            for driver_id, exp_pts in zip(
                race_result["driver_id"].values,
                race_result["sim_expected_points"].values,
            ):
                if driver_id in driver_total_points:
                    driver_total_points[driver_id] += exp_pts

        # Build championship table
        rows: List[Dict] = []
        for driver_id, total_pts in driver_total_points.items():
            rows.append({
                "driver_id": driver_id,
                "expected_total_points": total_pts,
                "points_std": 0.0,
            })

        return pd.DataFrame(rows).sort_values("expected_total_points", ascending=False)


def run_simulation(
    predictor: object,
    feature_matrix: pd.DataFrame,
    season: int,
    race_round: int,
    circuit_type: str = "mixed",
    n_simulations: int = 10000,
) -> pd.DataFrame:
    """
    Convenience function: predict + simulate a specific race.

    Args:
        predictor: Trained F1Predictor instance
        feature_matrix: Full feature matrix
        season: Year
        race_round: Round number
        circuit_type: Circuit classification
        n_simulations: Number of Monte Carlo runs

    Returns:
        Simulation results DataFrame
    """
    race_data = feature_matrix[
        (feature_matrix["season"] == season)
        & (feature_matrix["round"] == race_round)
    ]

    if race_data.empty:
        logger.warning("No data for %d round %d", season, race_round)
        return pd.DataFrame()

    predictions = predictor.predict_race(race_data)

    simulator = RaceSimulator(n_simulations=n_simulations)
    results = simulator.simulate_race(predictions, circuit_type)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from data.models.predictor import F1Predictor

    predictor = F1Predictor()
    predictor.load()

    feature_matrix = pd.read_parquet("data/cache/processed/feature_matrix.parquet")

    latest_season = int(feature_matrix["season"].max())
    latest_round = int(
        feature_matrix[feature_matrix["season"] == latest_season]["round"].max()
    )

    print(f"Simulating {latest_season} Round {latest_round}...")
    results = run_simulation(predictor, feature_matrix, latest_season, latest_round)

    if not results.empty:
        display_cols = [
            "driver_id",
            "predicted_position",
            "sim_win_pct",
            "sim_podium_pct",
            "sim_expected_points",
            "sim_median_position",
        ]
        available = [c for c in display_cols if c in results.columns]
        print(results[available].head(20).to_string())
