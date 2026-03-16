"""
Regulation Change Modeling — causal features for F1 rule changes.

F1 performance is heavily influenced by regulation changes. This module
explicitly models these regime shifts rather than letting ELO slowly adapt.

Features:
- Regulation era identification and distance metrics
- Structural break detection in constructor performance
- Budget cap era features
- Regulation reset factors for ELO system
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Major regulation changes with estimated impact magnitude (0-1)
# Higher magnitude = more performance reset expected
REGULATION_CHANGES: Dict[int, Dict] = {
    1966: {"magnitude": 0.5, "type": "engine", "desc": "3.0L engine formula"},
    1983: {"magnitude": 0.4, "type": "aero", "desc": "Flat bottom rules"},
    1989: {"magnitude": 0.5, "type": "engine", "desc": "Turbo ban, 3.5L NA"},
    1994: {"magnitude": 0.4, "type": "safety", "desc": "Post-Senna safety reforms"},
    1998: {"magnitude": 0.5, "type": "aero", "desc": "Grooved tyres, narrower cars"},
    2005: {"magnitude": 0.3, "type": "aero", "desc": "Reduced aero, single tyre rule"},
    2006: {"magnitude": 0.5, "type": "engine", "desc": "V8 2.4L engines"},
    2009: {"magnitude": 0.7, "type": "aero", "desc": "Aero overhaul + KERS"},
    2011: {"magnitude": 0.3, "type": "aero", "desc": "DRS + Pirelli tyres"},
    2014: {"magnitude": 0.8, "type": "engine", "desc": "Hybrid V6 turbo PU"},
    2017: {"magnitude": 0.5, "type": "aero", "desc": "Wider cars, more downforce"},
    2019: {"magnitude": 0.3, "type": "aero", "desc": "Simplified front wing"},
    2022: {"magnitude": 0.85, "type": "aero", "desc": "Ground effect + budget cap"},
    2026: {"magnitude": 0.8, "type": "engine", "desc": "New PU regs + active aero"},
}

# Regulatory eras — periods of stable regulations
REGULATION_ERAS: List[Dict] = [
    {"id": 1, "start": 1950, "end": 1965, "name": "Early Formula", "engine": "Various", "budget_cap": False},
    {"id": 2, "start": 1966, "end": 1982, "name": "3L Era", "engine": "3.0L", "budget_cap": False},
    {"id": 3, "start": 1983, "end": 1988, "name": "Turbo Era", "engine": "1.5L Turbo", "budget_cap": False},
    {"id": 4, "start": 1989, "end": 1993, "name": "NA Era", "engine": "3.5L NA", "budget_cap": False},
    {"id": 5, "start": 1994, "end": 1997, "name": "Post-Senna", "engine": "3.0L V10", "budget_cap": False},
    {"id": 6, "start": 1998, "end": 2005, "name": "Grooved Tyre Era", "engine": "3.0L V10", "budget_cap": False},
    {"id": 7, "start": 2006, "end": 2008, "name": "V8 Era", "engine": "2.4L V8", "budget_cap": False},
    {"id": 8, "start": 2009, "end": 2013, "name": "KERS Era", "engine": "2.4L V8 + KERS", "budget_cap": False},
    {"id": 9, "start": 2014, "end": 2016, "name": "Hybrid V6 Era I", "engine": "1.6L V6 Turbo Hybrid", "budget_cap": False},
    {"id": 10, "start": 2017, "end": 2021, "name": "Wide Car Era", "engine": "1.6L V6 Turbo Hybrid", "budget_cap": False},
    {"id": 11, "start": 2022, "end": 2025, "name": "Ground Effect Era", "engine": "1.6L V6 Turbo Hybrid", "budget_cap": True},
    {"id": 12, "start": 2026, "end": 2030, "name": "Active Aero Era", "engine": "New PU", "budget_cap": True},
]


def get_regulation_era(season: int) -> Dict:
    """Get the regulation era for a given season."""
    for era in REGULATION_ERAS:
        if era["start"] <= season <= era["end"]:
            return era
    return REGULATION_ERAS[-1]


def is_regulation_change(season: int) -> bool:
    """Check if a season is a regulation change year."""
    return season in REGULATION_CHANGES


def regulation_magnitude(season: int) -> float:
    """Get the regulation change magnitude for a season (0 if no change)."""
    change = REGULATION_CHANGES.get(season)
    return change["magnitude"] if change else 0.0


def regulation_distance(season: int) -> int:
    """
    Years since the last major regulation change.

    Returns 0 in a change year, 1 the year after, etc.
    Teams typically converge over time within stable regulations.
    """
    change_years = sorted(REGULATION_CHANGES.keys())
    for year in reversed(change_years):
        if season >= year:
            return season - year
    return season - 1950  # before any tracked changes


def regulation_convergence_factor(season: int) -> float:
    """
    How converged the field is expected to be (0 = just after reg change, 1 = fully converged).

    Uses an exponential decay: convergence = 1 - exp(-distance / tau)
    Tau = 3 years (field typically converges within 3-4 years)
    """
    dist = regulation_distance(season)
    tau = 3.0
    return 1.0 - np.exp(-dist / tau)


def compute_elo_reset_factors(season: int) -> Dict[str, float]:
    """
    Compute ELO reset factors for a regulation change year.

    Constructor ELO is reset more aggressively than driver ELO
    because regulations primarily affect car performance.

    Returns:
        Dict with 'driver' and 'constructor' reset factors (0 = full reset, 1 = no reset)
    """
    if season not in REGULATION_CHANGES:
        return {"driver": 1.0, "constructor": 1.0}

    change = REGULATION_CHANGES[season]
    magnitude = change["magnitude"]

    # Constructors reset harder — regs affect car more than driver skill
    constructor_retain = 1.0 - magnitude
    # Drivers retain more — talent persists across regs (but team change matters)
    driver_retain = 1.0 - (magnitude * 0.4)

    return {
        "driver": round(driver_retain, 3),
        "constructor": round(constructor_retain, 3),
    }


def detect_structural_breaks(
    race_results: pd.DataFrame,
    window: int = 5,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Detect structural breaks in constructor performance using CUSUM-inspired approach.

    A structural break occurs when a team's average finishing position shifts
    significantly (e.g., due to car upgrade, regulation adaptation, or key hire).

    Args:
        race_results: DataFrame with season, round, constructor_id, position columns
        window: Rolling window size for moving average
        threshold: Z-score threshold for detecting a break

    Returns:
        DataFrame with columns: season, round, constructor_id, break_magnitude,
        break_direction ('up' = improved, 'down' = worsened)
    """
    breaks = []

    for cid, group in race_results.groupby("constructor_id"):
        group = group.sort_values(["season", "round"])
        positions = group["position"].values

        if len(positions) < window * 2:
            continue

        # Rolling statistics
        rolling_mean = pd.Series(positions).rolling(window, min_periods=window).mean()
        rolling_std = pd.Series(positions).rolling(window, min_periods=window).std()

        for i in range(window, len(positions)):
            if pd.isna(rolling_mean.iloc[i]) or pd.isna(rolling_std.iloc[i]):
                continue
            if rolling_std.iloc[i] == 0:
                continue

            # Compare current value to rolling average
            z_score = (positions[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]

            if abs(z_score) > threshold:
                row = group.iloc[i]
                breaks.append({
                    "season": int(row["season"]),
                    "round": int(row["round"]),
                    "constructor_id": cid,
                    "break_magnitude": abs(z_score),
                    "break_direction": "up" if z_score < 0 else "down",  # lower position = better
                    "pre_mean": rolling_mean.iloc[i],
                    "current_position": positions[i],
                })

    return pd.DataFrame(breaks)


def compute_constructor_regime_score(
    race_results: pd.DataFrame,
    season: int,
    constructor_id: str,
    lookback_races: int = 10,
) -> float:
    """
    Compute a regime change score for a constructor at a given point in time.

    Measures how much the constructor's recent performance deviates from
    their longer-term average. High positive = improving, high negative = declining.

    Args:
        race_results: Historical results
        season: Current season
        constructor_id: Constructor to evaluate
        lookback_races: Number of recent races to compare

    Returns:
        Score in positions (negative = improving, positive = declining)
    """
    cid_results = race_results[
        race_results["constructor_id"] == constructor_id
    ].sort_values(["season", "round"])

    # Filter to before current season
    historical = cid_results[cid_results["season"] < season]

    if len(historical) < lookback_races * 2:
        return 0.0

    recent = historical.tail(lookback_races)["position"].mean()
    longer_term = historical.tail(lookback_races * 3)["position"].mean()

    return recent - longer_term  # negative = recent is better


def build_regulation_features(season: int, constructor_id: Optional[str] = None) -> Dict[str, float]:
    """
    Build regulation-related features for a single row in the feature matrix.

    Args:
        season: Championship year
        constructor_id: Optional constructor for constructor-specific features

    Returns:
        Dict of feature_name -> value
    """
    era = get_regulation_era(season)

    features = {
        "reg_change_year": float(is_regulation_change(season)),
        "reg_magnitude": regulation_magnitude(season),
        "reg_distance": float(regulation_distance(season)),
        "reg_convergence": regulation_convergence_factor(season),
        "reg_era_id": float(era["id"]),
        "reg_budget_cap": float(era.get("budget_cap", False)),
        "reg_is_engine_change": float(
            REGULATION_CHANGES.get(season, {}).get("type") == "engine"
        ),
        "reg_is_aero_change": float(
            REGULATION_CHANGES.get(season, {}).get("type") == "aero"
        ),
    }

    return features


if __name__ == "__main__":
    # Demo: show regulation features for notable seasons
    for season in [2009, 2014, 2017, 2022, 2023, 2024, 2025, 2026]:
        features = build_regulation_features(season)
        era = get_regulation_era(season)
        print(f"\n{season} ({era['name']}):")
        for k, v in features.items():
            print(f"  {k}: {v}")

        reset = compute_elo_reset_factors(season)
        if reset["constructor"] < 1.0:
            print(f"  ELO reset — driver retain: {reset['driver']}, constructor retain: {reset['constructor']}")
