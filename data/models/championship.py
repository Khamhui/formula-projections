"""
Championship Monte Carlo — simulates remaining season with scenario analysis.

Extends the base RaceSimulator.simulate_championship() with:
- Per-race championship probability trajectories
- What-if scenario analysis (forced outcomes)
- Clinch analysis (earliest possible clinch round)
- Sprint race points (2021+)
- WCC (World Constructors Championship) simulation

Usage:
    python -m data.models.championship --season 2025
    python -m data.models.championship --season 2025 --scenario "max_verstappen:DNF:5"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.models.simulator import RaceSimulator, POINTS

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache" / "processed"

# Sprint race points (2023+ format)
SPRINT_POINTS: Dict[int, int] = {
    1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1,
}

# Fastest lap bonus (for drivers finishing in points)
FASTEST_LAP_BONUS = 1


class ChampionshipSimulator:
    """Full-season championship Monte Carlo with scenario analysis."""

    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.simulator = RaceSimulator(n_simulations=n_simulations, random_seed=random_seed)
        self.rng = np.random.default_rng(random_seed)

    def simulate_season(
        self,
        race_predictions: List[pd.DataFrame],
        circuit_types: List[str],
        race_names: List[str],
        current_standings: Optional[Dict[str, float]] = None,
        constructor_standings: Optional[Dict[str, float]] = None,
        conditions_per_race: Optional[List[str]] = None,
        constructor_map: Optional[Dict[str, str]] = None,
        sprint_races: Optional[List[bool]] = None,
        scenarios: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> Dict:
        """
        Simulate remaining championship with per-race trajectory tracking.

        Args:
            race_predictions: Prediction DataFrames per remaining race
            circuit_types: Circuit type per race
            race_names: Human-readable race names (e.g. "Bahrain GP")
            current_standings: driver_id -> current WDC points
            constructor_standings: constructor_id -> current WCC points
            conditions_per_race: "dry"/"wet"/"mixed" per race
            constructor_map: driver_id -> constructor_id
            sprint_races: Boolean list, True if race has sprint
            scenarios: Forced outcomes — driver_id -> {race_index: outcome}
                       outcome can be "DNF", or a position number like "1", "3"

        Returns:
            Dict with:
                - wdc: DataFrame of WDC results (expected points, win%, podium%)
                - wcc: DataFrame of WCC results (if constructor_map provided)
                - trajectory: Dict of driver_id -> list of championship win% per race
                - points_percentiles: Dict of driver_id -> {p5, p25, p50, p75, p95}
                - clinch: Dict of driver_id -> earliest possible clinch round
        """
        if not race_predictions:
            return {"wdc": pd.DataFrame(), "trajectory": {}, "clinch": {}}

        current_standings = current_standings or {}
        constructor_standings = constructor_standings or {}
        conditions_per_race = conditions_per_race or ["dry"] * len(race_predictions)
        sprint_races = sprint_races or [False] * len(race_predictions)
        scenarios = scenarios or {}

        # Collect all unique drivers
        all_drivers = _collect_unique_drivers(race_predictions)
        n_total = len(all_drivers)
        driver_idx = {d: i for i, d in enumerate(all_drivers)}
        n_sims = self.n_simulations

        # Cumulative points matrix: (n_sims, n_drivers)
        cum_points = np.zeros((n_sims, n_total))
        for did, pts in current_standings.items():
            if did in driver_idx:
                cum_points[:, driver_idx[did]] += pts

        # Trajectory tracking: per-race championship win probability
        trajectory: Dict[str, List[float]] = {d: [] for d in all_drivers}
        n_races = len(race_predictions)

        for race_idx in range(n_races):
            pred = race_predictions[race_idx]
            ct = circuit_types[race_idx]
            cond = conditions_per_race[race_idx]
            is_sprint = sprint_races[race_idx]

            # Apply scenarios: force specific outcomes
            pred = self._apply_scenario(pred, scenarios, race_idx)

            # Simulate race points
            race_points = self.simulator._simulate_race_points(
                pred, ct, cond, constructor_map,
            )

            for did, pts_array in race_points.items():
                if did in driver_idx:
                    cum_points[:, driver_idx[did]] += pts_array

            # Sprint race simulation (simplified — use same predictions with sprint points)
            if is_sprint:
                sprint_points = self._simulate_sprint_points(
                    pred, ct, cond, constructor_map,
                )
                for did, pts_array in sprint_points.items():
                    if did in driver_idx:
                        cum_points[:, driver_idx[did]] += pts_array

            # Fastest lap bonus (random driver in points gets +1)
            fl_bonus = self._simulate_fastest_lap(n_sims, n_total, cum_points)
            cum_points += fl_bonus

            # Track trajectory: championship win% after each race
            ranks = (-cum_points).argsort(axis=1).argsort(axis=1) + 1
            for did in all_drivers:
                idx = driver_idx[did]
                win_pct = float((ranks[:, idx] == 1).sum()) / n_sims * 100
                trajectory[did].append(win_pct)

            logger.debug(
                "Championship sim: race %d/%d (%s) complete",
                race_idx + 1, n_races, race_names[race_idx] if race_idx < len(race_names) else "?",
            )

        # Final championship results
        final_ranks = (-cum_points).argsort(axis=1).argsort(axis=1) + 1

        wdc_rows = []
        points_percentiles = {}
        for did in all_drivers:
            idx = driver_idx[did]
            pts = cum_points[:, idx]
            rnk = final_ranks[:, idx]

            pcts = {
                "p5": float(np.percentile(pts, 5)),
                "p25": float(np.percentile(pts, 25)),
                "p50": float(np.percentile(pts, 50)),
                "p75": float(np.percentile(pts, 75)),
                "p95": float(np.percentile(pts, 95)),
            }
            points_percentiles[did] = pcts

            wdc_rows.append({
                "driver_id": did,
                "expected_total_points": float(pts.mean()),
                "points_std": float(pts.std()),
                "points_median": pcts["p50"],
                "championship_win_pct": float((rnk == 1).sum()) / n_sims * 100,
                "top3_pct": float((rnk <= 3).sum()) / n_sims * 100,
                "top5_pct": float((rnk <= 5).sum()) / n_sims * 100,
            })

        wdc_df = pd.DataFrame(wdc_rows).sort_values("expected_total_points", ascending=False)

        # WCC simulation
        wcc_df = pd.DataFrame()
        if constructor_map:
            wcc_df = self._compute_wcc(
                cum_points, all_drivers, driver_idx, constructor_map, constructor_standings,
            )

        # Clinch analysis
        clinch = self._clinch_analysis(
            cum_points, all_drivers, driver_idx, n_races, race_predictions,
        )

        return {
            "wdc": wdc_df,
            "wcc": wcc_df,
            "trajectory": trajectory,
            "points_percentiles": points_percentiles,
            "clinch": clinch,
            "race_names": race_names,
        }

    def _apply_scenario(
        self,
        pred: pd.DataFrame,
        scenarios: Dict[str, Dict[int, str]],
        race_idx: int,
    ) -> pd.DataFrame:
        """Apply forced scenario outcomes to a prediction DataFrame."""
        pred = pred.copy()
        for driver_id, race_outcomes in scenarios.items():
            if race_idx not in race_outcomes:
                continue
            outcome = race_outcomes[race_idx]
            mask = pred["driver_id"] == driver_id
            if not mask.any():
                continue
            if outcome.upper() == "DNF":
                pred.loc[mask, "predicted_position"] = 25.0
                pred.loc[mask, "prob_dnf"] = 0.99
            else:
                try:
                    forced_pos = float(outcome)
                    pred.loc[mask, "predicted_position"] = forced_pos
                    pred.loc[mask, "prob_dnf"] = 0.0
                except ValueError:
                    pass
        return pred

    def _simulate_sprint_points(
        self,
        predictions: pd.DataFrame,
        circuit_type: str,
        conditions: str,
        constructor_map: Optional[Dict[str, str]],
    ) -> Dict[str, np.ndarray]:
        """Simulate sprint race and return points per driver."""
        finish_positions, dnf_flags, _, driver_ids = self.simulator._run_simulation_core(
            predictions, circuit_type, conditions, constructor_map,
        )

        n_sims = self.n_simulations
        n_drivers = len(predictions)

        # Sprint points lookup
        sprint_lookup = np.array([SPRINT_POINTS.get(p, 0) for p in range(n_drivers + 1)])
        sprint_pts = sprint_lookup[finish_positions]
        sprint_pts[dnf_flags] = 0

        result: Dict[str, np.ndarray] = {}
        for i, did in enumerate(driver_ids):
            result[str(did)] = sprint_pts[:, i]
        return result

    def _simulate_fastest_lap(
        self,
        n_sims: int,
        n_drivers: int,
        cum_points: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate fastest lap bonus — random driver in points gets +1.

        Vectorized: sort all sims at once, then use cumulative weight
        thresholds to pick a weighted-random top-10 driver per sim.
        """
        bonus = np.zeros_like(cum_points)

        # Get top-10 indices for all sims at once: (n_sims, 10)
        k = min(10, n_drivers)
        top10 = np.argsort(-cum_points, axis=1)[:, :k]

        # Weights: front-runners more likely to get fastest lap
        base_weights = np.array([3, 3, 2, 2, 1, 1, 1, 1, 1, 1][:k], dtype=float)
        cum_weights = np.cumsum(base_weights / base_weights.sum())

        # Draw random values and find which weight bucket they fall into
        draws = self.rng.random(n_sims)
        slot_indices = np.searchsorted(cum_weights, draws)
        slot_indices = np.clip(slot_indices, 0, k - 1)

        # Map slot indices to actual driver indices
        chosen = top10[np.arange(n_sims), slot_indices]
        bonus[np.arange(n_sims), chosen] += FASTEST_LAP_BONUS

        return bonus

    def _compute_wcc(
        self,
        cum_points: np.ndarray,
        all_drivers: List[str],
        driver_idx: Dict[str, int],
        constructor_map: Dict[str, str],
        constructor_standings: Dict[str, float],
    ) -> pd.DataFrame:
        """Aggregate driver points into constructor championship results."""
        constructors = set(constructor_map.values())
        n_sims = self.n_simulations

        constructor_points = np.zeros((n_sims, len(constructors)))
        cid_list = sorted(constructors)
        cid_idx = {c: i for i, c in enumerate(cid_list)}

        # Add current standings
        for cid, pts in constructor_standings.items():
            if cid in cid_idx:
                constructor_points[:, cid_idx[cid]] += pts

        # Sum simulated race points per constructor (exclude pre-seeded current standings
        # since those are already accounted for via constructor_standings above)
        current_standings_by_driver = {}
        for did in all_drivers:
            if did in driver_idx:
                # The initial cum_points for this driver included their current standings
                current_standings_by_driver[did] = float(cum_points[0, driver_idx[did]]) if cum_points[0, driver_idx[did]] > 0 else 0.0

        for did in all_drivers:
            cid = constructor_map.get(did)
            if cid and cid in cid_idx and did in driver_idx:
                # Add simulated points only (total minus the pre-seeded current standings)
                simulated_only = cum_points[:, driver_idx[did]] - current_standings_by_driver.get(did, 0.0)
                constructor_points[:, cid_idx[cid]] += simulated_only

        # Rank
        constructor_ranks = (-constructor_points).argsort(axis=1).argsort(axis=1) + 1

        rows = []
        for cid in cid_list:
            idx = cid_idx[cid]
            pts = constructor_points[:, idx]
            rnk = constructor_ranks[:, idx]
            rows.append({
                "constructor_id": cid,
                "expected_total_points": float(pts.mean()),
                "points_std": float(pts.std()),
                "championship_win_pct": float((rnk == 1).sum()) / n_sims * 100,
                "top3_pct": float((rnk <= 3).sum()) / n_sims * 100,
            })

        return pd.DataFrame(rows).sort_values("expected_total_points", ascending=False)

    def _clinch_analysis(
        self,
        cum_points: np.ndarray,
        all_drivers: List[str],
        driver_idx: Dict[str, int],
        n_remaining: int,
        race_predictions: List[pd.DataFrame],
    ) -> Dict[str, Optional[int]]:
        """
        Determine the earliest round each driver could mathematically clinch.

        A driver clinches when their minimum possible remaining points
        exceed any rival's maximum possible remaining points.
        """
        max_points_per_race = 25 + 8 + 1  # win + sprint win + fastest lap = 34

        clinch = {}
        for did in all_drivers:
            idx = driver_idx[did]
            current = float(cum_points[0, idx])  # all sims start equal
            max_current = current + n_remaining * max_points_per_race

            # Find max rival points possible
            max_rival = 0.0
            for rival in all_drivers:
                if rival == did:
                    continue
                rival_idx = driver_idx[rival]
                rival_current = float(cum_points[0, rival_idx])
                rival_max = rival_current + n_remaining * max_points_per_race
                max_rival = max(max_rival, rival_max)

            # Can this driver clinch? Only possible if they're ahead enough
            # that even with max rival points, they can't be caught
            if current > max_rival:
                clinch[did] = 0  # already clinched
            else:
                # Find earliest round where clinch is possible
                points_needed = max_rival - current
                if max_points_per_race > 0:
                    min_races_needed = int(np.ceil(points_needed / max_points_per_race))
                    if min_races_needed <= n_remaining:
                        clinch[did] = min_races_needed
                    else:
                        clinch[did] = None  # can't clinch mathematically
                else:
                    clinch[did] = None

        return clinch

    def what_if(
        self,
        race_predictions: List[pd.DataFrame],
        circuit_types: List[str],
        race_names: List[str],
        scenario: Dict[str, Dict[int, str]],
        **kwargs,
    ) -> Dict:
        """
        Convenience wrapper for scenario analysis.

        Args:
            scenario: e.g. {"max_verstappen": {0: "DNF", 1: "DNF"}, "norris": {0: "1"}}
                      Keys are driver_ids, values are {race_index: outcome}

        Returns:
            Same as simulate_season() but with forced outcomes applied
        """
        return self.simulate_season(
            race_predictions, circuit_types, race_names,
            scenarios=scenario, **kwargs,
        )


def _collect_unique_drivers(predictions: List[pd.DataFrame]) -> List[str]:
    """Collect all unique drivers across all race predictions, preserving order."""
    seen = set()
    drivers = []
    for pred in predictions:
        for d in pred["driver_id"].values:
            if d not in seen:
                drivers.append(d)
                seen.add(d)
    return drivers


def remaining_calendar(season: int) -> List[Dict]:
    """
    Get remaining races in the calendar using FastF1 schedule.

    Returns list of dicts with: round, name, circuit_id, date, is_sprint
    """
    try:
        import fastf1
        schedule = fastf1.get_event_schedule(season, include_testing=False)

        from datetime import datetime
        now = datetime.now()

        remaining = []
        for _, event in schedule.iterrows():
            event_date = pd.Timestamp(event.get("EventDate", event.get("Session5DateUtc")))
            if event_date and event_date > pd.Timestamp(now):
                remaining.append({
                    "round": int(event.get("RoundNumber", 0)),
                    "name": event.get("EventName", "Unknown"),
                    "circuit_id": event.get("Location", "unknown").lower().replace(" ", "_"),
                    "date": str(event_date.date()),
                    "is_sprint": bool(event.get("EventFormat", "") == "sprint_shootout"),
                })

        return remaining
    except Exception as e:
        logger.warning("Could not load FastF1 schedule: %s", e)
        return []


def load_current_standings(season: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Load current WDC and WCC standings from cached data.

    Returns:
        (driver_standings, constructor_standings) as dicts of id -> points
    """
    driver_standings = {}
    constructor_standings = {}

    standings_path = CACHE_DIR / "driver_standings.parquet"
    if standings_path.exists():
        df = pd.read_parquet(standings_path)
        season_df = df[df["season"] == season]
        if not season_df.empty:
            latest_round = season_df["round"].max()
            latest = season_df[season_df["round"] == latest_round]
            for _, row in latest.iterrows():
                driver_standings[row["driver_id"]] = float(row.get("points", 0))

    constructor_path = CACHE_DIR / "constructor_standings.parquet"
    if constructor_path.exists():
        df = pd.read_parquet(constructor_path)
        season_df = df[df["season"] == season]
        if not season_df.empty:
            latest_round = season_df["round"].max()
            latest = season_df[season_df["round"] == latest_round]
            for _, row in latest.iterrows():
                constructor_standings[row["constructor_id"]] = float(row.get("points", 0))

    return driver_standings, constructor_standings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Championship Monte Carlo")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument(
        "--scenario", type=str, nargs="*",
        help="Forced outcomes: 'driver_id:outcome:race_idx' (e.g. 'max_verstappen:DNF:0')",
    )
    parser.add_argument("--sims", type=int, default=10000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Parse scenarios
    scenarios = {}
    if args.scenario:
        for s in args.scenario:
            parts = s.split(":")
            if len(parts) == 3:
                driver_id, outcome, race_idx = parts
                scenarios.setdefault(driver_id, {})[int(race_idx)] = outcome

    # Load calendar
    calendar = remaining_calendar(args.season)
    if not calendar:
        print(f"No remaining races found for {args.season}")
        print("Ensure FastF1 is installed: pip install fastf1")
        exit(1)

    print(f"\n=== {args.season} Championship Monte Carlo ===")
    print(f"Remaining races: {len(calendar)}")
    for race in calendar:
        sprint_tag = " [SPRINT]" if race["is_sprint"] else ""
        print(f"  R{race['round']:02d} {race['name']} ({race['date']}){sprint_tag}")

    # Load standings
    driver_standings, constructor_standings = load_current_standings(args.season)
    if driver_standings:
        print(f"\nCurrent standings loaded: {len(driver_standings)} drivers")
    else:
        print("\nNo current standings found — starting from zero")

    # Load predictor and generate predictions for remaining races
    try:
        from data.models.predictor import F1Predictor
        from data.features.engineer import prepare_training_data

        model = F1Predictor()
        model.load()

        fm = pd.read_parquet(CACHE_DIR / "feature_matrix.parquet")

        # For remaining races, use the latest feature data as proxy
        latest_season = fm[fm["season"] == args.season]
        latest_round = int(latest_season["round"].max()) if not latest_season.empty else 0

        # Generate predictions for each remaining race using latest available features
        race_predictions = []
        circuit_types_list = []
        race_names_list = []
        sprint_list = []

        for race in calendar:
            race_data = fm[
                (fm["season"] == args.season) & (fm["round"] == latest_round)
            ]
            if race_data.empty:
                continue

            pred = model.predict_race(race_data)
            race_predictions.append(pred)
            circuit_types_list.append("mixed")  # default
            race_names_list.append(race["name"])
            sprint_list.append(race["is_sprint"])

        if not race_predictions:
            print("No predictions could be generated")
            exit(1)

        # Build constructor map from feature matrix
        constructor_map_data = {}
        if "constructor_id" in fm.columns:
            latest = fm[(fm["season"] == args.season) & (fm["round"] == latest_round)]
            for _, row in latest.iterrows():
                constructor_map_data[row["driver_id"]] = row["constructor_id"]

        # Run simulation
        sim = ChampionshipSimulator(n_simulations=args.sims)
        result = sim.simulate_season(
            race_predictions=race_predictions,
            circuit_types=circuit_types_list,
            race_names=race_names_list,
            current_standings=driver_standings,
            constructor_standings=constructor_standings,
            constructor_map=constructor_map_data,
            sprint_races=sprint_list,
            scenarios=scenarios if scenarios else None,
        )

        # Display WDC results
        wdc = result["wdc"]
        if not wdc.empty:
            print(f"\n=== WDC Projections ({args.sims:,} simulations) ===")
            print(f"{'Driver':<25} {'Exp Pts':>8} {'Win%':>6} {'Top3%':>6} {'Median':>7}")
            print("-" * 60)
            for _, row in wdc.head(20).iterrows():
                print(
                    f"{row['driver_id']:<25} {row['expected_total_points']:>8.1f} "
                    f"{row['championship_win_pct']:>5.1f}% {row['top3_pct']:>5.1f}% "
                    f"{row['points_median']:>7.1f}"
                )

        # Display WCC results
        wcc = result.get("wcc")
        if wcc is not None and not wcc.empty:
            print(f"\n=== WCC Projections ===")
            print(f"{'Constructor':<25} {'Exp Pts':>8} {'Win%':>6} {'Top3%':>6}")
            print("-" * 50)
            for _, row in wcc.head(10).iterrows():
                print(
                    f"{row['constructor_id']:<25} {row['expected_total_points']:>8.1f} "
                    f"{row['championship_win_pct']:>5.1f}% {row['top3_pct']:>5.1f}%"
                )

        # Save results
        wdc.to_csv(CACHE_DIR / f"championship_wdc_{args.season}.csv", index=False)
        print(f"\nResults saved to {CACHE_DIR / f'championship_wdc_{args.season}.csv'}")

    except Exception as e:
        logger.error("Championship simulation failed: %s", e)
        raise
