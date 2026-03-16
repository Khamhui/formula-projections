"""
In-Race Live Prediction Engine — lap-by-lap probability updates.

Combines pre-race model predictions with real-time race state from OpenF1
to produce updated win/podium/points probabilities during the race.

The approach: Bayesian-inspired update where pre-race priors are adjusted
based on in-race evidence (current position, gaps, tire state, track status).
"""

from __future__ import annotations

import collections
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RaceState:
    """Snapshot of the current race state for all drivers."""

    def __init__(self):
        self.lap: int = 0
        self.total_laps: int = 0
        self.track_status: str = "clear"  # clear, sc, vsc, red
        self.drivers: Dict[str, DriverState] = {}

    @property
    def laps_remaining(self) -> int:
        return max(0, self.total_laps - self.lap)

    @property
    def race_progress(self) -> float:
        if self.total_laps == 0:
            return 0.0
        return self.lap / self.total_laps


class DriverState:
    """Current race state for a single driver."""

    def __init__(self, driver_id: str):
        self.driver_id = driver_id
        self.position: int = 0
        self.gap_to_leader: float = 0.0
        self.gap_to_ahead: float = 0.0
        self.tire_compound: str = "unknown"  # soft, medium, hard, intermediate, wet
        self.tire_age: int = 0
        self.pits_completed: int = 0
        self.last_lap_time: float = 0.0
        self.best_lap_time: float = 0.0
        self.is_in_pit: bool = False
        self.is_retired: bool = False
        self.sector1: float = 0.0
        self.sector2: float = 0.0
        self.sector3: float = 0.0


class LiveRacePredictor:
    """
    Updates predictions during a live race using pre-race model + in-race state.

    Strategy:
    1. Start with pre-race predictions as prior
    2. As race progresses, weight shifts toward in-race evidence
    3. Current position becomes dominant predictor late in race
    4. Safety car / VSC increases uncertainty
    """

    def __init__(
        self,
        pre_race_predictions: pd.DataFrame,
        total_laps: int = 57,
    ):
        """
        Args:
            pre_race_predictions: DataFrame from F1Predictor.predict_race()
                Must have: driver_id, predicted_position, prob_winner, prob_podium
            total_laps: Expected race distance in laps
        """
        self.pre_race = pre_race_predictions.set_index("driver_id")
        self.total_laps = total_laps
        self.current_state: Optional[RaceState] = None
        self.prediction_history: collections.deque = collections.deque(maxlen=200)

    def update(self, race_state: RaceState) -> pd.DataFrame:
        """
        Generate updated predictions from current race state.

        Args:
            race_state: Current snapshot of the race

        Returns:
            DataFrame with updated predictions per driver
        """
        self.current_state = race_state
        progress = race_state.race_progress

        # Evidence weight increases with race progress
        # Early race: trust pre-race model more
        # Late race: trust current positions more
        evidence_weight = self._evidence_weight(progress, race_state.track_status)
        prior_weight = 1.0 - evidence_weight

        results = []
        n_drivers = len(race_state.drivers)

        for driver_id, state in race_state.drivers.items():
            if state.is_retired:
                results.append({
                    "driver_id": driver_id,
                    "live_position": state.position,
                    "predicted_position": float(n_drivers + 1),
                    "win_prob": 0.0,
                    "podium_prob": 0.0,
                    "points_prob": 0.0,
                    "is_retired": True,
                    "tire_compound": state.tire_compound,
                    "tire_age": state.tire_age,
                    "pits_completed": state.pits_completed,
                    "gap_to_leader": state.gap_to_leader,
                })
                continue

            # Pre-race prior (if available)
            if driver_id in self.pre_race.index:
                prior_pos = self.pre_race.loc[driver_id, "predicted_position"]
                prior_win = self.pre_race.loc[driver_id].get("prob_winner", 0.05)
                prior_podium = self.pre_race.loc[driver_id].get("prob_podium", 0.15)
            else:
                prior_pos = 10.0
                prior_win = 1.0 / max(n_drivers, 1)
                prior_podium = 3.0 / max(n_drivers, 1)

            # In-race evidence
            evidence_pos = float(state.position)

            # Pace-adjusted position (factor in recent lap times vs field)
            pace_adjustment = self._pace_adjustment(state, race_state)
            adjusted_evidence_pos = evidence_pos + pace_adjustment

            # Tire strategy adjustment
            tire_adjustment = self._tire_strategy_adjustment(state, race_state)
            adjusted_evidence_pos += tire_adjustment

            # Bayesian-ish combination
            predicted_pos = prior_weight * prior_pos + evidence_weight * adjusted_evidence_pos

            # Probability updates
            win_prob = self._compute_live_win_prob(
                state, race_state, prior_win, evidence_weight,
            )
            podium_prob = self._compute_live_podium_prob(
                state, race_state, prior_podium, evidence_weight,
            )
            points_prob = self._compute_live_points_prob(
                state, race_state, evidence_weight,
            )

            results.append({
                "driver_id": driver_id,
                "live_position": state.position,
                "predicted_position": predicted_pos,
                "win_prob": win_prob,
                "podium_prob": podium_prob,
                "points_prob": points_prob,
                "is_retired": False,
                "tire_compound": state.tire_compound,
                "tire_age": state.tire_age,
                "pits_completed": state.pits_completed,
                "gap_to_leader": state.gap_to_leader,
                "pace_adjustment": pace_adjustment,
            })

        df = pd.DataFrame(results).sort_values("predicted_position")

        # Normalize probabilities
        active_mask = ~df["is_retired"]
        if active_mask.any():
            # Win probs must sum to 1.0
            win_total = df.loc[active_mask, "win_prob"].sum()
            if win_total > 0:
                df.loc[active_mask, "win_prob"] /= win_total

            # Podium probs must sum to min(3, n_active)
            n_active = active_mask.sum()
            podium_total = df.loc[active_mask, "podium_prob"].sum()
            target_sum = min(3.0, float(n_active))
            if podium_total > 0:
                df.loc[active_mask, "podium_prob"] *= target_sum / podium_total
                df["podium_prob"] = df["podium_prob"].clip(0, 1)

        # Record history
        self.prediction_history.append({
            "lap": race_state.lap,
            "predictions": df.to_dict("records"),
        })

        return df

    def _evidence_weight(self, progress: float, track_status: str) -> float:
        """
        How much to weight in-race evidence vs pre-race prior.

        Uses sigmoid curve: slow start, then rapid increase mid-race.
        Safety car reduces evidence weight (positions are compressed).
        """
        # Sigmoid: 0.1 at lap 0, 0.5 at 40% race, 0.95 at 80% race
        weight = 1.0 / (1.0 + np.exp(-10 * (progress - 0.4)))

        # Safety car reduces evidence weight (positions are misleading)
        if track_status == "sc":
            weight *= 0.7
        elif track_status == "vsc":
            weight *= 0.85
        elif track_status == "red":
            weight *= 0.5

        return float(np.clip(weight, 0.05, 0.98))

    def _pace_adjustment(self, driver: DriverState, race: RaceState) -> float:
        """
        Adjust predicted position based on pace vs field.

        Faster than average = likely to gain positions (negative adjustment).
        Slower = likely to lose (positive adjustment).
        """
        if driver.last_lap_time <= 0:
            return 0.0

        # Field median lap time
        lap_times = [
            d.last_lap_time for d in race.drivers.values()
            if d.last_lap_time > 0 and not d.is_retired and not d.is_in_pit
        ]
        if not lap_times:
            return 0.0

        median_time = np.median(lap_times)
        if median_time <= 0:
            return 0.0

        # Pace delta as fraction of median (positive = slower)
        pace_delta = (driver.last_lap_time - median_time) / median_time

        # Scale: 1% faster ≈ 0.5 position gain potential
        return pace_delta * 50.0

    def _tire_strategy_adjustment(self, driver: DriverState, race: RaceState) -> float:
        """
        Adjust for tire strategy — old tires on a hard compound = likely to lose pace.
        Fresh tires = likely to gain.
        """
        # Degradation curves (positions lost per lap on old tires)
        deg_rates = {
            "soft": 0.08,      # degrades fastest
            "medium": 0.05,
            "hard": 0.03,
            "intermediate": 0.04,
            "wet": 0.02,
            "unknown": 0.05,
        }

        rate = deg_rates.get(driver.tire_compound, 0.05)

        # Only penalize significantly after ~15 laps (soft), ~25 (medium), ~35 (hard)
        sweet_spot = {"soft": 15, "medium": 25, "hard": 35}.get(driver.tire_compound, 20)
        excess_laps = max(0, driver.tire_age - sweet_spot)

        return excess_laps * rate

    def _compute_live_win_prob(
        self,
        driver: DriverState,
        race: RaceState,
        prior: float,
        evidence_weight: float,
    ) -> float:
        """Compute live win probability."""
        if driver.is_retired:
            return 0.0

        # Position-based evidence (exponential decay from P1)
        pos_factor = np.exp(-0.8 * (driver.position - 1))

        # Gap penalty (if gap to leader > 20s, win is very unlikely)
        if driver.gap_to_leader > 0:
            gap_factor = np.exp(-driver.gap_to_leader / 15.0)
        else:
            gap_factor = 1.0

        evidence_prob = pos_factor * gap_factor

        # Combine prior and evidence
        combined = (1 - evidence_weight) * prior + evidence_weight * evidence_prob

        return float(np.clip(combined, 0.001, 0.99))

    def _compute_live_podium_prob(
        self,
        driver: DriverState,
        race: RaceState,
        prior: float,
        evidence_weight: float,
    ) -> float:
        """Compute live podium probability."""
        if driver.is_retired:
            return 0.0

        # Position-based: P1-P3 high, drops off quickly
        if driver.position <= 3:
            pos_factor = 0.9 - 0.1 * (driver.position - 1)
        else:
            pos_factor = np.exp(-0.5 * (driver.position - 3))

        # Gap to P3
        p3_drivers = [d for d in race.drivers.values() if d.position == 3 and not d.is_retired]
        if p3_drivers and driver.position > 3:
            gap_to_p3 = driver.gap_to_leader - p3_drivers[0].gap_to_leader
            if gap_to_p3 > 0:
                pos_factor *= np.exp(-gap_to_p3 / 10.0)

        evidence_prob = pos_factor
        combined = (1 - evidence_weight) * prior + evidence_weight * evidence_prob

        return float(np.clip(combined, 0.001, 0.99))

    def _compute_live_points_prob(
        self,
        driver: DriverState,
        race: RaceState,
        evidence_weight: float,
    ) -> float:
        """Compute live points probability (P1-P10)."""
        if driver.is_retired:
            return 0.0

        if driver.position <= 10:
            evidence_prob = 0.95 - 0.03 * (driver.position - 1)
        else:
            evidence_prob = np.exp(-0.3 * (driver.position - 10))

        prior = 0.5
        combined = (1 - evidence_weight) * prior + evidence_weight * evidence_prob

        return float(np.clip(combined, 0.01, 0.99))
