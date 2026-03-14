"""
F1 ELO Rating System — adapted from chess for motorsport.

Unlike tennis (1v1), F1 has 20 drivers per race. We use a multi-player ELO
approach: each race generates (n*(n-1))/2 pairwise comparisons.

Multiple ELO variants:
- Overall driver ELO
- Circuit-type ELO (street, high-speed, technical, mixed)
- Wet-weather ELO
- Qualifying ELO (separate from race pace)
- Constructor ELO
- Head-to-head ELO (between teammates)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Circuit type classification
CIRCUIT_TYPES = {
    # Street circuits — low grip, tight corners, walls
    "monaco": "street", "marina_bay": "street", "baku": "street",
    "jeddah": "street", "albert_park": "street", "vegas": "street",
    "miami": "street",
    # High-speed — long straights, power-dependent
    "monza": "high_speed", "spa": "high_speed", "silverstone": "high_speed",
    "red_bull_ring": "high_speed", "bahrain": "high_speed",
    # Technical — lots of corners, downforce-dependent
    "hungaroring": "technical", "catalunya": "technical",
    "suzuka": "technical", "interlagos": "technical", "imola": "technical",
    "zandvoort": "technical", "losail": "technical",
    # Mixed
    "shanghai": "mixed", "americas": "mixed", "yas_marina": "mixed",
    "sepang": "mixed", "hockenheimring": "mixed", "nurburgring": "mixed",
    "istanbul": "mixed", "portimao": "mixed", "mugello": "mixed",
    "sakhir": "mixed", "ricard": "mixed",
}

DEFAULT_CIRCUIT_TYPE = "mixed"

# K-factors control how much ratings change per race
K_FACTOR = 6          # Base K for race results
K_QUALIFYING = 4      # Lower K for qualifying (less variance)
K_WET = 8             # Higher K for wet (reveals more skill delta)
K_CONSTRUCTOR = 4     # Constructors change slower


_DEFAULT_RATING = 1500.0


@dataclass
class EloRating:
    """An ELO rating with history tracking."""
    rating: float = _DEFAULT_RATING
    peak: float = _DEFAULT_RATING
    matches: int = 0
    track_history: bool = True
    history: list = field(default_factory=list)

    def update(self, delta: float):
        self.rating += delta
        self.peak = max(self.peak, self.rating)
        self.matches += 1
        if self.track_history:
            self.history.append(self.rating)


class F1EloSystem:
    """
    Multi-dimensional ELO system for F1.

    Tracks separate ratings for:
    - Overall race performance
    - Circuit type specialization (street/high-speed/technical)
    - Wet weather ability
    - Qualifying pace
    - Constructor performance
    """

    def __init__(self, k_factor: float = K_FACTOR):
        self.k = k_factor

        # Driver ratings: driver_id -> EloRating
        self.overall: dict[str, EloRating] = {}
        self.by_circuit_type: dict[str, dict[str, EloRating]] = {
            ct: {} for ct in ["street", "high_speed", "technical", "mixed"]
        }
        self.wet: dict[str, EloRating] = {}
        self.qualifying: dict[str, EloRating] = {}

        # Constructor ratings
        self.constructors: dict[str, EloRating] = {}

        # Head-to-head between teammates
        self.h2h: dict[str, dict[str, EloRating]] = {}

        # Processed races for tracking
        self.races_processed: list[dict] = []

    def _get_or_create(self, store: dict, key: str, initial: float = 1500.0) -> EloRating:
        if key not in store:
            store[key] = EloRating(rating=initial)
        return store[key]

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected probability of A finishing ahead of B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _process_pairwise(
        self,
        results: list[tuple[str, int]],
        rating_store: dict[str, EloRating],
        k: float,
    ) -> dict[str, float]:
        """
        Process a multi-player result as pairwise comparisons.

        Args:
            results: List of (driver_id, finishing_position) tuples
            rating_store: Which ELO dict to update
            k: K-factor for this comparison

        Returns:
            Dict of driver_id -> rating delta
        """
        n = len(results)
        if n < 2:
            return {}

        deltas: dict[str, float] = {driver_id: 0.0 for driver_id, _ in results}

        for i in range(n):
            for j in range(i + 1, n):
                id_a, pos_a = results[i]
                id_b, pos_b = results[j]

                rating_a = self._get_or_create(rating_store, id_a)
                rating_b = self._get_or_create(rating_store, id_b)

                expected_a = self._expected_score(rating_a.rating, rating_b.rating)

                # Actual score: 1 if A finished ahead, 0 if behind, 0.5 if tied
                if pos_a < pos_b:
                    actual_a = 1.0
                elif pos_a > pos_b:
                    actual_a = 0.0
                else:
                    actual_a = 0.5

                # Scale K by number of comparisons to keep total update reasonable
                scaled_k = k / (n - 1)
                delta = scaled_k * (actual_a - expected_a)

                deltas[id_a] += delta
                deltas[id_b] -= delta

        # Apply deltas
        for driver_id, delta in deltas.items():
            self._get_or_create(rating_store, driver_id).update(delta)

        return deltas

    def process_race(
        self,
        season: int,
        race_round: int,
        circuit_id: str,
        results: pd.DataFrame,
        is_wet: bool = False,
    ):
        """
        Process a race result and update all ELO ratings.

        Args:
            season: Year
            race_round: Round number
            circuit_id: Circuit identifier
            results: DataFrame with columns: driver_id, constructor_id, position, grid
            is_wet: Whether the race was wet
        """
        if results.empty:
            return

        # Filter to classified finishers
        classified = results.dropna(subset=["position"])
        if classified.empty:
            return

        driver_results = [
            (row["driver_id"], int(row["position"]))
            for _, row in classified.iterrows()
        ]

        # 1. Overall ELO
        self._process_pairwise(driver_results, self.overall, self.k)

        # 2. Circuit-type ELO
        circuit_type = CIRCUIT_TYPES.get(circuit_id, DEFAULT_CIRCUIT_TYPE)
        ct_store = self.by_circuit_type[circuit_type]
        self._process_pairwise(driver_results, ct_store, self.k)

        # 3. Wet ELO (only update in wet races — more weight)
        if is_wet:
            self._process_pairwise(driver_results, self.wet, K_WET)

        # 4. Constructor ELO — use best-placed driver per constructor
        constructor_results = (
            classified.groupby("constructor_id")["position"]
            .min()
            .reset_index()
            .sort_values("position")
        )
        constructor_pairs = [
            (row["constructor_id"], int(row["position"]))
            for _, row in constructor_results.iterrows()
        ]
        self._process_pairwise(constructor_pairs, self.constructors, K_CONSTRUCTOR)

        # 5. Teammate head-to-head
        for constructor_id, group in classified.groupby("constructor_id"):
            if len(group) >= 2:
                teammates = group.sort_values("position")
                d1 = teammates.iloc[0]["driver_id"]
                d2 = teammates.iloc[1]["driver_id"]
                pair_key = tuple(sorted([d1, d2]))

                if pair_key[0] not in self.h2h:
                    self.h2h[pair_key[0]] = {}
                h2h_store = self.h2h[pair_key[0]]

                h2h_results = [
                    (d1, int(teammates.iloc[0]["position"])),
                    (d2, int(teammates.iloc[1]["position"])),
                ]
                self._process_pairwise(h2h_results, h2h_store, self.k)

        self.races_processed.append({
            "season": season,
            "round": race_round,
            "circuit_id": circuit_id,
            "circuit_type": circuit_type,
            "is_wet": is_wet,
            "n_drivers": len(driver_results),
        })

    def process_qualifying(self, results: pd.DataFrame):
        """Process qualifying results to update qualifying ELO."""
        if results.empty:
            return

        driver_results = [
            (row["driver_id"], int(row["position"]))
            for _, row in results.iterrows()
            if pd.notna(row.get("position"))
        ]

        self._process_pairwise(driver_results, self.qualifying, K_QUALIFYING)

    def get_driver_ratings(self) -> pd.DataFrame:
        """Get current ratings for all drivers across all dimensions."""
        rows = []

        for driver_id in self.overall:
            overall = self.overall[driver_id]
            row = {
                "driver_id": driver_id,
                "elo_overall": overall.rating,
                "elo_peak": overall.peak,
                "elo_matches": overall.matches,
                "elo_qualifying": self.qualifying[driver_id].rating if driver_id in self.qualifying else _DEFAULT_RATING,
                "elo_wet": self.wet[driver_id].rating if driver_id in self.wet else _DEFAULT_RATING,
            }

            for ct in ["street", "high_speed", "technical", "mixed"]:
                ct_store = self.by_circuit_type[ct]
                row[f"elo_{ct}"] = ct_store[driver_id].rating if driver_id in ct_store else _DEFAULT_RATING

            rows.append(row)

        return pd.DataFrame(rows).sort_values("elo_overall", ascending=False)

    def get_constructor_ratings(self) -> pd.DataFrame:
        """Get current constructor ratings."""
        rows = [
            {
                "constructor_id": cid,
                "elo": r.rating,
                "peak": r.peak,
                "matches": r.matches,
            }
            for cid, r in self.constructors.items()
        ]
        return pd.DataFrame(rows).sort_values("elo", ascending=False)

    def get_driver_history(self, driver_id: str) -> list[float]:
        """Get rating history for a specific driver."""
        rating = self.overall.get(driver_id)
        return rating.history if rating else []

    def get_matchup_prediction(self, driver_a: str, driver_b: str) -> dict:
        """Predict head-to-head probability between two drivers."""
        ra = self.overall.get(driver_a, EloRating())
        rb = self.overall.get(driver_b, EloRating())

        prob_a = self._expected_score(ra.rating, rb.rating)

        return {
            "driver_a": driver_a,
            "driver_b": driver_b,
            "prob_a_wins": round(prob_a, 4),
            "prob_b_wins": round(1 - prob_a, 4),
            "elo_diff": round(ra.rating - rb.rating, 1),
            "rating_a": round(ra.rating, 1),
            "rating_b": round(rb.rating, 1),
        }

    def snapshot(self) -> dict[str, pd.DataFrame]:
        """Take a snapshot of all current ratings."""
        return {
            "drivers": self.get_driver_ratings(),
            "constructors": self.get_constructor_ratings(),
        }


def build_elo_from_history(
    race_results: pd.DataFrame,
    qualifying: Optional[pd.DataFrame] = None,
    wet_races: Optional[set[tuple[int, int]]] = None,
) -> F1EloSystem:
    """
    Build ELO ratings from historical race data.

    Args:
        race_results: DataFrame with columns: season, round, circuit_id,
                      driver_id, constructor_id, position, grid
        qualifying: Optional qualifying data
        wet_races: Optional set of (season, round) tuples for wet races

    Returns:
        Fully computed F1EloSystem
    """
    elo = F1EloSystem()
    wet_races = wet_races or set()

    # Process chronologically
    race_results = race_results.sort_values(["season", "round"])

    for (season, rnd), group in race_results.groupby(["season", "round"]):
        circuit_id = group.iloc[0]["circuit_id"]
        is_wet = (season, rnd) in wet_races

        elo.process_race(
            season=int(season),
            race_round=int(rnd),
            circuit_id=circuit_id,
            results=group,
            is_wet=is_wet,
        )

    # Process qualifying if available
    if qualifying is not None and not qualifying.empty:
        for (season, rnd), group in qualifying.groupby(["season", "round"]):
            elo.process_qualifying(group)

    logger.info(
        f"ELO built from {len(elo.races_processed)} races, "
        f"{len(elo.overall)} drivers, {len(elo.constructors)} constructors"
    )

    return elo
