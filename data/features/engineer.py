"""
Feature engineering for F1 race prediction.

Combines Jolpica historical data, FastF1 granular data, and ELO ratings
into a feature matrix suitable for XGBoost training.

Target: finishing position (or binary: podium/no podium, points/no points)

Feature categories:
1. ELO ratings (overall, circuit-type, qualifying, wet, constructor)
2. Recent form (last N races)
3. Circuit-specific performance
4. Grid position and qualifying pace (Q1/Q2/Q3 times)
5. Constructor strength
6. Head-to-head vs teammate
7. Season momentum
8. Weather conditions (auto-detected from FastF1)
9. Tire strategy indicators (FastF1, 2018+)
10. Telemetry-derived features (FastF1, 2018+)
11. Sprint race performance (2021+)
12. Pit stop strategy (2012+)
13. Track status performance (SC/VSC gains, 2022+)
14. Constructor development trajectory (season-over-season)
15. Practice session pace (FP1/FP2/FP3 from FastF1, 2018+)
16. DNF risk indicators (mechanical reliability, crash history)
17. Championship standings (WDC/WCC position, points gap, title contention)
18. Teammate comparison (ELO diff, qualifying H2H, positions gained)
19. Momentum / form (exponentially weighted position, form vs season avg)
20. FP2 long run analysis (race pace indicator, tire degradation rate)
21. Grid × circuit type interaction (grid impact varies by circuit type)
22. Overtaking difficulty (historical positions gained per circuit)
23. Constructor upgrade detection (pace jumps mid-season)
24. Cold start / rookie handling (career races, teammate ELO prior)
25. Pit stop performance (team pit stop quality from FastF1 stint data)
"""

import bisect
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from data.features.elo import F1EloSystem, CIRCUIT_TYPES, DEFAULT_CIRCUIT_TYPE
from data.features.regulation import (
    build_regulation_features,
    compute_elo_reset_factors,
    REGULATION_CHANGES,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "cache" / "processed"

# Pre-compiled regexes for lap time parsing (called thousands of times)
_RE_MM_SS = re.compile(r'^(\d+):(\d+\.\d+)$')
_RE_SS = re.compile(r'^(\d+\.\d+)$')

# FastF1 feature columns used in rolling averages
_FASTF1_ROLLING_COLS = [
    "lap_consistency", "lap_mean", "lap_best",
    "sector1_consistency", "sector2_consistency", "sector3_consistency",
    "sector1_mean", "sector2_mean", "sector3_mean",
    "speedst_avg", "speedst_max", "speedfl_avg",
    "tire_deg_avg", "tire_deg_worst",
    "n_stints", "n_compounds",
]

_WINDOWS = [3, 5, 10, 20]

# FastF1 TrackStatus codes
# 1=AllClear, 2=Yellow, 4=SafetyCar, 5=RedFlag, 6=VSC, 7=VSCEnding
_SC_CODES = {"4", "41", "42", "45", "46", "47"}
_VSC_CODES = {"6", "61", "67", "671"}
_YELLOW_CODES = {"2", "21", "24", "26", "12"}
_SC_VSC_CODES = _SC_CODES | _VSC_CODES  # Pre-computed union

# Circuit type sets for interaction features (derived from CIRCUIT_TYPES)
_STREET_CIRCUITS = {cid for cid, ct in CIRCUIT_TYPES.items() if ct == "street"}
_HIGH_SPEED_CIRCUITS = {cid for cid, ct in CIRCUIT_TYPES.items() if ct == "high_speed"}
_TECHNICAL_CIRCUITS = {cid for cid, ct in CIRCUIT_TYPES.items() if ct == "technical"}


def _linear_trend(values: np.ndarray) -> float:
    """Slope of a linear fit over a sequence of values."""
    return np.polyfit(range(len(values)), values, 1)[0]


def _ewm(values: Iterable[float], alpha: float) -> float:
    """Exponentially weighted moving average. Recent values weighted more.

    Caller must ensure values is non-empty.
    """
    it = iter(values)
    ewm = next(it)
    for v in it:
        ewm = alpha * v + (1 - alpha) * ewm
    return ewm


def _is_classified(status: str) -> bool:
    """Whether a driver was classified (finished or lapped)."""
    return status == "Finished" or status.startswith("+")


def _parse_lap_time(time_str) -> Optional[float]:
    """Parse F1 lap time string (e.g. '1:28.586') to seconds."""
    if not time_str or pd.isna(time_str):
        return None
    time_str = str(time_str).strip()
    match = _RE_MM_SS.match(time_str)
    if match:
        return int(match.group(1)) * 60 + float(match.group(2))
    match = _RE_SS.match(time_str)
    if match:
        return float(match.group(1))
    return None


def _default_circuit_features() -> dict:
    return {
        "circuit_races": 0,
        "circuit_avg_pos": np.nan,
        "circuit_best_pos": np.nan,
        "circuit_podium_rate": 0,
        "circuit_ewm_pos": np.nan,
        "circuit_avg_gained": np.nan,
        "circuit_recent_gained": np.nan,
        "circuit_win_streak": 0,
        "circuit_quali_avg": np.nan,
        "circuit_quali_best": np.nan,
        "circuit_quali_recent": np.nan,
    }


def _default_season_features() -> dict:
    return {
        "season_races": 0,
        "season_avg_pos": np.nan,
        "season_points_total": 0,
        "season_trend": 0.0,
    }


def _default_rolling_features() -> dict:
    features = {}
    for w in _WINDOWS:
        features[f"pos_last{w}_mean"] = np.nan
        features[f"pos_last{w}_std"] = np.nan
        features[f"pos_last{w}_best"] = np.nan
        features[f"podium_rate_last{w}"] = 0
        features[f"points_rate_last{w}"] = 0
        features[f"dnf_rate_last{w}"] = 0
    return features


# Precompute defaults once (used repeatedly in the hot loop)
_DEFAULT_CIRCUIT = _default_circuit_features()
_DEFAULT_SEASON = _default_season_features()
_DEFAULT_ROLLING = _default_rolling_features()
_DEFAULT_ALL = {**_DEFAULT_ROLLING, **_DEFAULT_CIRCUIT, **_DEFAULT_SEASON}


def _build_driver_code_map(race_results: pd.DataFrame) -> dict[str, str]:
    """Build mapping from 3-letter driver codes (VER, HAM) to driver IDs."""
    if "driver_code" not in race_results.columns:
        return {}
    pairs = race_results[["driver_code", "driver_id"]].dropna().drop_duplicates()
    return dict(zip(pairs["driver_code"], pairs["driver_id"]))


def _detect_wet_races(fastf1_laps: pd.DataFrame) -> set[tuple[int, int]]:
    """Auto-detect wet races from FastF1 weather data."""
    if fastf1_laps is None or fastf1_laps.empty:
        return set()
    if "rainfall" not in fastf1_laps.columns:
        return set()

    race_laps = fastf1_laps[fastf1_laps["session_type"] == "R"]
    if race_laps.empty:
        return set()

    wet = set()
    for (year, gp), group in race_laps.groupby(["year", "gp"]):
        if group["rainfall"].any():
            wet.add((int(year), int(gp)))

    if wet:
        logger.info(f"Auto-detected {len(wet)} wet races from FastF1 weather data")
    return wet


def _compute_fastf1_race_stats(
    fastf1_laps: pd.DataFrame,
    driver_code_map: dict[str, str],
) -> dict[tuple[int, int, str], dict]:
    """
    Pre-compute per-driver per-race stats from FastF1 lap data.

    Returns dict keyed by (season, round, driver_id) -> stats dict.
    Used to build rolling FastF1 features without data leakage.
    """
    if fastf1_laps is None or fastf1_laps.empty:
        return {}

    race_laps = fastf1_laps[fastf1_laps["session_type"] == "R"]
    if race_laps.empty:
        return {}

    stats = {}

    for (year, gp), group in race_laps.groupby(["year", "gp"]):
        for driver_code, dlaps in group.groupby("Driver"):
            driver_id = driver_code_map.get(driver_code)
            if not driver_id:
                continue

            accurate = dlaps[dlaps["IsAccurate"]] if "IsAccurate" in dlaps.columns else dlaps
            if accurate.empty:
                continue

            row = {}

            # Lap time consistency
            if "LapTime_s" in accurate.columns:
                times = accurate["LapTime_s"].dropna()
                if len(times) > 3:
                    row["lap_consistency"] = times.std()
                    row["lap_mean"] = times.mean()
                    row["lap_best"] = times.min()

            # Sector consistency
            for s in [1, 2, 3]:
                col = f"Sector{s}Time_s"
                if col in accurate.columns:
                    vals = accurate[col].dropna()
                    if len(vals) > 3:
                        row[f"sector{s}_consistency"] = vals.std()
                        row[f"sector{s}_mean"] = vals.mean()

            # Speed trap stats (most predictive: straight-line speed)
            for speed_col in ["SpeedST", "SpeedFL"]:
                if speed_col in accurate.columns:
                    vals = accurate[speed_col].dropna()
                    if not vals.empty:
                        row[f"{speed_col.lower()}_avg"] = vals.mean()
                        row[f"{speed_col.lower()}_max"] = vals.max()

            # Tire degradation: lap time trend per stint
            if "Stint" in accurate.columns and "LapTime_s" in accurate.columns:
                deg_slopes = []
                for _, stint_laps in accurate.groupby("Stint"):
                    times = stint_laps["LapTime_s"].dropna().values
                    if len(times) >= 5:
                        deg_slopes.append(_linear_trend(times))
                if deg_slopes:
                    row["tire_deg_avg"] = np.mean(deg_slopes)
                    row["tire_deg_worst"] = max(deg_slopes)

            # Strategy indicators
            if "Stint" in accurate.columns:
                row["n_stints"] = accurate["Stint"].nunique()
            if "Compound" in accurate.columns:
                row["n_compounds"] = accurate["Compound"].nunique()

            if row:
                stats[(int(year), int(gp), driver_id)] = row

    logger.info(f"Computed FastF1 stats for {len(stats)} driver-race entries")
    return stats


def _build_quali_times_index(
    qualifying: pd.DataFrame,
) -> dict[tuple[int, int], dict[str, dict]]:
    """
    Pre-compute qualifying times in seconds, indexed by (season, round) -> driver_id -> stats.
    Converts Q1/Q2/Q3 strings to seconds and computes per-race field stats.
    """
    if qualifying is None or qualifying.empty:
        return {}

    index = {}
    for (season, rnd), group in qualifying.groupby(["season", "round"]):
        drivers = {}
        all_q3 = []
        all_q2 = []
        all_q1 = []

        for _, row in group.iterrows():
            driver_id = row["driver_id"]
            q1 = _parse_lap_time(row.get("q1"))
            q2 = _parse_lap_time(row.get("q2"))
            q3 = _parse_lap_time(row.get("q3"))

            # Handle duplicate entries (regular Q + sprint Q mixed):
            # keep the row with the fastest best time
            best_new = min(filter(None, [q1, q2, q3]), default=None)
            if driver_id in drivers:
                existing = drivers[driver_id]
                best_old = min(filter(None, [existing["q1_s"], existing["q2_s"], existing["q3_s"]]), default=None)
                if best_old is not None and (best_new is None or best_new >= best_old):
                    continue  # keep existing (faster)

            drivers[driver_id] = {"q1_s": q1, "q2_s": q2, "q3_s": q3}

            if q1 is not None:
                all_q1.append(q1)
            if q2 is not None:
                all_q2.append(q2)
            if q3 is not None:
                all_q3.append(q3)

        # Compute field medians for relative pace
        median_q3 = np.median(all_q3) if all_q3 else None
        median_q2 = np.median(all_q2) if all_q2 else None
        median_q1 = np.median(all_q1) if all_q1 else None
        # Best available median (prefer Q3 > Q2 > Q1)
        best_median = median_q3 or median_q2 or median_q1

        for driver_id, times in drivers.items():
            times["best_field_median"] = best_median

        index[(int(season), int(rnd))] = drivers

    logger.info(f"Built qualifying times index for {len(index)} races")
    return index


def _build_sprint_index(
    sprints: pd.DataFrame,
) -> dict[tuple[int, int], dict[str, dict]]:
    """Index sprint results by (season, round) -> driver_id -> stats."""
    if sprints is None or sprints.empty:
        return {}

    index = {}
    for (season, rnd), group in sprints.groupby(["season", "round"]):
        drivers = {}
        for _, row in group.iterrows():
            pos = row.get("position")
            status = row.get("status", "")
            drivers[row["driver_id"]] = {
                "sprint_position": pos,
                "sprint_grid": int(row.get("grid", 0)),
                "sprint_points": float(row.get("points", 0)),
                "sprint_dnf": 0 if (pos is not None and _is_classified(status)) else 1,
            }
        index[(int(season), int(rnd))] = drivers

    logger.info(f"Built sprint index for {len(index)} sprint races")
    return index


def _compute_fastf1_weather(
    fastf1_laps: pd.DataFrame,
) -> dict[tuple[int, int], dict]:
    """Pre-compute per-race weather conditions from FastF1."""
    if fastf1_laps is None or fastf1_laps.empty:
        return {}

    weather_cols = ["air_temp_avg", "track_temp_avg", "humidity_avg", "wind_speed_avg"]
    available = [c for c in weather_cols if c in fastf1_laps.columns]
    if not available:
        return {}

    race_laps = fastf1_laps[fastf1_laps["session_type"] == "R"]
    if race_laps.empty:
        return {}

    weather = {}
    for (year, gp), group in race_laps.groupby(["year", "gp"]):
        w = {}
        for col in available:
            vals = group[col].dropna()
            if not vals.empty:
                w[col] = vals.iloc[0]  # session-level aggregate already
        if w:
            weather[(int(year), int(gp))] = w

    logger.info(f"Built weather index for {len(weather)} races")
    return weather


def _compute_track_status_stats(
    fastf1_laps: pd.DataFrame,
    driver_code_map: dict[str, str],
) -> dict[tuple[int, int, str], dict]:
    """
    Compute per-driver SC/VSC performance stats from FastF1 TrackStatus.

    Measures how many positions a driver gains/loses during SC/VSC periods.
    """
    if fastf1_laps is None or fastf1_laps.empty:
        return {}
    if "TrackStatus" not in fastf1_laps.columns or "Position" not in fastf1_laps.columns:
        return {}

    race_laps = fastf1_laps[fastf1_laps["session_type"] == "R"]
    if race_laps.empty:
        return {}

    stats = {}
    for (year, gp), group in race_laps.groupby(["year", "gp"]):
        for driver_code, dlaps in group.groupby("Driver"):
            driver_id = driver_code_map.get(driver_code)
            if not driver_id:
                continue

            dlaps_sorted = dlaps.sort_values("LapNumber")
            track_statuses = dlaps_sorted["TrackStatus"].astype(str)
            positions = dlaps_sorted["Position"]

            # Count laps under different conditions
            sc_laps = track_statuses.isin(_SC_CODES).sum()
            vsc_laps = track_statuses.isin(_VSC_CODES).sum()
            yellow_laps = track_statuses.isin(_YELLOW_CODES).sum()
            total_laps = len(dlaps_sorted)

            row = {
                "sc_laps_pct": sc_laps / total_laps if total_laps > 0 else 0,
                "vsc_laps_pct": vsc_laps / total_laps if total_laps > 0 else 0,
                "yellow_laps_pct": yellow_laps / total_laps if total_laps > 0 else 0,
            }

            # Position change during SC/VSC periods
            if not positions.empty and len(positions) > 1:
                pos_changes = positions.diff()

                sc_mask = track_statuses.isin(_SC_VSC_CODES)
                if sc_mask.any():
                    sc_pos_changes = pos_changes[sc_mask].dropna()
                    if not sc_pos_changes.empty:
                        # Negative = gained positions (moved up)
                        row["sc_pos_gain"] = -sc_pos_changes.sum()

            if row:
                stats[(int(year), int(gp), driver_id)] = row

    logger.info(f"Computed track status stats for {len(stats)} driver-race entries")
    return stats


def _compute_practice_pace(
    fastf1_laps: pd.DataFrame,
    driver_code_map: dict,
) -> tuple[dict, dict]:
    """
    Extract practice session pace from FastF1 (FP1/FP2/FP3).
    Returns dict keyed by (year, gp, driver_id) -> pace stats.
    """
    if fastf1_laps is None or fastf1_laps.empty:
        return {}, {}

    fp_laps = fastf1_laps[fastf1_laps["session_type"].isin(["FP1", "FP2", "FP3"])]
    if fp_laps.empty:
        return {}, {}

    stats = {}
    for (year, gp, session), group in fp_laps.groupby(["year", "gp", "session_type"]):
        for driver_code, dlaps in group.groupby("Driver"):
            driver_id = driver_code_map.get(driver_code)
            if not driver_id:
                continue

            if "LapTime_s" not in dlaps.columns:
                continue

            accurate = dlaps[dlaps["IsAccurate"]] if "IsAccurate" in dlaps.columns else dlaps
            times = accurate["LapTime_s"].dropna()
            if len(times) < 3:
                continue

            key = (int(year), int(gp), driver_id)
            if key not in stats:
                stats[key] = {"fp_best": [], "fp_mean": [], "fp_laps": 0}

            stats[key]["fp_best"].append(times.min())
            stats[key]["fp_mean"].append(times.mean())
            stats[key]["fp_laps"] += len(times)

            # FP2 long run analysis: consecutive laps on same compound, >= 6 laps
            if session == "FP2" and "Compound" in accurate.columns:
                sorted_laps = accurate.sort_values("LapNumber")
                long_run_times = []
                current_compound = None
                current_run = []

                for _, lap in sorted_laps.iterrows():
                    compound = lap.get("Compound")
                    lap_time = lap.get("LapTime_s")
                    if pd.isna(lap_time):
                        # Break on missing time
                        if len(current_run) >= 6:
                            long_run_times.extend(current_run)
                        current_run = []
                        current_compound = None
                        continue
                    if compound == current_compound:
                        current_run.append(lap_time)
                    else:
                        if len(current_run) >= 6:
                            long_run_times.extend(current_run)
                        current_run = [lap_time]
                        current_compound = compound

                # Flush last run
                if len(current_run) >= 6:
                    long_run_times.extend(current_run)

                if long_run_times:
                    stats[key]["fp2_long_run_pace"] = np.mean(long_run_times)
                    # Degradation rate: linear trend slope over long run laps
                    if len(long_run_times) >= 6:
                        stats[key]["fp2_deg_rate"] = _linear_trend(np.array(long_run_times))

    # Flatten: take best across all FP sessions + pre-compute field medians
    result = {}
    race_bests = {}  # (year, gp) -> list of best times for field median
    for key, s in stats.items():
        best = min(s["fp_best"])
        entry = {
            "fp_best_time": best,
            "fp_mean_time": np.mean(s["fp_mean"]),
            "fp_total_laps": s["fp_laps"],
        }
        # FP2 long run stats (if available)
        if "fp2_long_run_pace" in s:
            entry["fp2_long_run_pace"] = s["fp2_long_run_pace"]
        if "fp2_deg_rate" in s:
            entry["fp2_deg_rate"] = s["fp2_deg_rate"]
        result[key] = entry
        race_key = (key[0], key[1])
        race_bests.setdefault(race_key, []).append(best)

    # Compute field medians once per race (not per driver)
    field_medians = {k: np.median(v) for k, v in race_bests.items()}

    if result:
        logger.info(f"Computed practice pace for {len(result)} driver-race entries")
    return result, field_medians


def _compute_cumulative_standings(
    race_results: pd.DataFrame,
    sprints: Optional[pd.DataFrame] = None,
) -> tuple[dict, dict]:
    """
    Derive per-round championship standings from race results and sprints.

    Unlike the API which only returns final-round standings, this computes
    cumulative standings *entering* each round — what a driver's championship
    position actually was before the race started.

    Returns:
        driver_standings_index: dict[(season, round), dict[driver_id, {position, points, wins}]]
        constructor_standings_index: dict[(season, round), dict[constructor_id, {position, points}]]
    """
    if race_results is None or race_results.empty:
        return {}, {}

    driver_standings_index: dict[tuple[int, int], dict[str, dict]] = {}
    constructor_standings_index: dict[tuple[int, int], dict[str, dict]] = {}

    # Index sprint points by (season, round, driver_id) for O(1) lookup
    sprint_pts: dict[tuple[int, int, str], float] = {}
    if sprints is not None and not sprints.empty:
        for (season, rnd), grp in sprints.groupby(["season", "round"]):
            for _, row in grp.iterrows():
                key = (int(season), int(rnd), row["driver_id"])
                sprint_pts[key] = sprint_pts.get(key, 0) + float(row.get("points", 0))

    for season, season_df in race_results.groupby("season"):
        season = int(season)
        rounds_sorted = sorted(season_df["round"].unique().astype(int))

        d_points: dict[str, float] = {}
        d_wins: dict[str, int] = {}
        c_points: dict[str, float] = {}

        for rnd in rounds_sorted:
            # Snapshot standings ENTERING this round (before processing)
            if d_points:
                sorted_d = sorted(
                    d_points.keys(),
                    key=lambda d: (-d_points[d], -d_wins.get(d, 0)),
                )
                driver_standings_index[(season, rnd)] = {
                    did: {
                        "position": pos,
                        "points": d_points[did],
                        "wins": d_wins.get(did, 0),
                    }
                    for pos, did in enumerate(sorted_d, 1)
                }
                sorted_c = sorted(c_points.keys(), key=lambda c: -c_points[c])
                constructor_standings_index[(season, rnd)] = {
                    cid: {"position": pos, "points": c_points[cid]}
                    for pos, cid in enumerate(sorted_c, 1)
                }

            # Accumulate this round's results
            round_data = season_df[season_df["round"] == rnd]
            for _, row in round_data.iterrows():
                did = row["driver_id"]
                cid = row.get("constructor_id", "unknown")
                pts = float(row.get("points", 0))
                pos = row.get("position")

                # Include sprint points for this round
                pts += sprint_pts.get((season, rnd, did), 0)

                d_points[did] = d_points.get(did, 0) + pts
                if pos is not None and not np.isnan(pos) and int(pos) == 1:
                    d_wins[did] = d_wins.get(did, 0) + 1
                c_points[cid] = c_points.get(cid, 0) + pts

    logger.info(
        f"Computed cumulative standings for {len(driver_standings_index)} race rounds"
    )
    return driver_standings_index, constructor_standings_index


def _compute_circuit_overtaking_rates(
    race_results: pd.DataFrame,
) -> dict[str, dict]:
    """
    Compute historical overtaking stats per circuit from race results.

    Returns dict keyed by circuit_id -> {
        "avg_positions_gained": float,  # average abs(grid - finish) per driver
        "grid_finish_map": dict[int, float],  # grid_pos -> avg finish position
    }
    """
    if race_results is None or race_results.empty:
        return {}

    stats = {}
    for circuit_id, group in race_results.groupby("circuit_id"):
        valid = group[
            (group["grid"] > 0)
            & group["position"].notna()
            & (group["position"] > 0)
        ]
        if valid.empty:
            continue

        deltas = (valid["grid"] - valid["position"]).abs()
        avg_delta = deltas.mean()

        # Per-grid-position average finish (for expected_grid_delta)
        grid_finish = {}
        for grid_pos, g_group in valid.groupby("grid"):
            grid_pos = int(grid_pos)
            if len(g_group) >= 3:
                grid_finish[grid_pos] = g_group["position"].mean()

        stats[circuit_id] = {
            "avg_positions_gained": avg_delta,
            "grid_finish_map": grid_finish,
        }

    logger.info(f"Computed overtaking rates for {len(stats)} circuits")
    return stats


def _compute_circuit_dna(
    race_results: pd.DataFrame,
) -> dict[str, dict]:
    """
    Compute circuit DNA — intrinsic characteristics of each circuit
    from historical race data.

    Returns dict keyed by circuit_id -> {
        grid_position_correlation: Spearman corr(grid, finish) — how much grid determines result
        avg_positions_changed: mean |grid - finish| — overtaking volume
        front_row_win_rate: P(win | grid <= 2)
        front_row_lock_rate: P(finish 1-2 | grid <= 2)
        attrition_rate: fraction of entries that DNF
        position_variance: std of finish positions
    }
    """
    if race_results is None or race_results.empty:
        return {}

    dna: dict[str, dict] = {}
    for circuit_id, group in race_results.groupby("circuit_id"):
        valid = group[
            (group["grid"] > 0)
            & group["position"].notna()
            & (group["position"] > 0)
        ]
        if len(valid) < 20:
            continue

        grids = valid["grid"].values.astype(float)
        finishes = valid["position"].values.astype(float)

        # Spearman rank correlation (handles ties and degenerate cases)
        corr, _ = spearmanr(grids, finishes)

        avg_delta = float(np.mean(np.abs(grids - finishes)))

        front_row = valid[valid["grid"] <= 2]
        front_row_win = float((front_row["position"] == 1).mean()) if len(front_row) > 0 else 0.0
        front_row_lock = float((front_row["position"] <= 2).mean()) if len(front_row) > 0 else 0.0

        # Attrition: fraction of all entries (including DNFs) that didn't finish
        classified = group["status"].apply(_is_classified)
        attrition = 1.0 - classified.mean()

        pos_variance = float(np.std(finishes))

        dna[circuit_id] = {
            "grid_position_correlation": float(corr) if np.isfinite(corr) else np.nan,
            "avg_positions_changed": avg_delta,
            "front_row_win_rate": front_row_win,
            "front_row_lock_rate": front_row_lock,
            "attrition_rate": float(attrition),
            "position_variance": pos_variance,
        }

    logger.info(f"Computed circuit DNA for {len(dna)} circuits")
    return dna


def _compute_pit_stop_stats(
    fastf1_laps: pd.DataFrame,
    driver_code_map: dict[str, str],
) -> dict[tuple[int, int, str], dict]:
    """
    Estimate pit stop performance from FastF1 stint transitions.

    Uses time between last lap of stint N and first lap of stint N+1
    as a proxy for pit stop duration (includes in-lap/out-lap delta).

    Returns dict keyed by (season, round, team) -> {pit_time_avg, pit_count, clean_pct}.
    """
    if fastf1_laps is None or fastf1_laps.empty:
        return {}

    race_laps = fastf1_laps[fastf1_laps["session_type"] == "R"]
    if race_laps.empty:
        return {}

    required_cols = {"Stint", "LapTime_s", "Team", "LapNumber"}
    if not required_cols.issubset(race_laps.columns):
        return {}

    stats = {}
    for (year, gp), group in race_laps.groupby(["year", "gp"]):
        team_pit_times: dict[str, list[float]] = {}

        for driver_code, dlaps in group.groupby("Driver"):
            team = dlaps["Team"].iloc[0] if "Team" in dlaps.columns else None
            if not team:
                continue

            sorted_laps = dlaps.sort_values("LapNumber")
            stints = sorted_laps["Stint"].values
            lap_times = sorted_laps["LapTime_s"].values

            # Detect pit stop laps: first lap of each new stint has inflated time
            for i in range(1, len(stints)):
                if stints[i] != stints[i - 1] and not np.isnan(lap_times[i]):
                    team_pit_times.setdefault(team, []).append(lap_times[i])

        for team, pit_times in team_pit_times.items():
            if not pit_times:
                continue
            median_pit = np.median(pit_times)
            # "Clean" pit stop: within 1.5x of median (no issues)
            clean = sum(1 for t in pit_times if t < median_pit * 1.5)
            stats[(int(year), int(gp), team)] = {
                "pit_time_avg": np.mean(pit_times),
                "pit_count": len(pit_times),
                "clean_pct": clean / len(pit_times),
            }

    logger.info(f"Computed pit stop stats for {len(stats)} team-race entries")
    return stats


def build_feature_matrix(
    race_results: pd.DataFrame,
    qualifying: pd.DataFrame,
    fastf1_laps: Optional[pd.DataFrame] = None,
    wet_races: Optional[set[tuple[int, int]]] = None,
    sprints: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the complete feature matrix for prediction.

    Each row = one driver in one race, with features computed from
    all data available BEFORE that race (no data leakage).

    Returns DataFrame with features + target column (position).
    """
    # Auto-detect wet races from FastF1 weather data
    auto_wet = _detect_wet_races(fastf1_laps)
    wet_races = (wet_races or set()) | auto_wet

    # Build driver code -> ID mapping for FastF1 merge
    driver_code_map = _build_driver_code_map(race_results)

    # Pre-compute all FastF1 per-race stats (keyed by season/round/driver_id)
    fastf1_stats = _compute_fastf1_race_stats(fastf1_laps, driver_code_map)
    has_fastf1 = bool(fastf1_stats)
    if has_fastf1:
        logger.info(f"FastF1 data available for {len(fastf1_stats)} driver-race entries")

    # Pre-compute qualifying times index
    quali_times_index = _build_quali_times_index(qualifying)

    # Pre-compute sprint index
    sprint_index = _build_sprint_index(sprints)
    has_sprints = bool(sprint_index)

    # Pre-compute weather index
    weather_index = _compute_fastf1_weather(fastf1_laps)

    # Pre-compute track status stats
    track_status_stats = _compute_track_status_stats(fastf1_laps, driver_code_map)

    # Pre-compute practice session pace
    practice_pace, fp_field_medians = _compute_practice_pace(fastf1_laps, driver_code_map)
    has_practice = bool(practice_pace)

    # Pre-compute championship standings index (derived from race results)
    driver_standings_index, constructor_standings_index = _compute_cumulative_standings(
        race_results, sprints
    )
    has_standings = bool(driver_standings_index)

    # Build max-round-per-season lookup for previous-season fallback
    standings_max_round: dict[int, int] = {}
    for (s, r) in driver_standings_index:
        if s not in standings_max_round or r > standings_max_round[s]:
            standings_max_round[s] = r

    # Pre-compute circuit overtaking rates (from all historical data)
    circuit_overtaking = _compute_circuit_overtaking_rates(race_results)

    # Pre-compute circuit DNA (grid correlation, attrition, etc.)
    circuit_dna = _compute_circuit_dna(race_results)

    # Pre-compute pit stop stats from FastF1 stint data
    pit_stop_stats = _compute_pit_stop_stats(fastf1_laps, driver_code_map)
    has_pit_stops = bool(pit_stop_stats)

    # Sort chronologically
    race_results = race_results.sort_values(["season", "round", "position"]).copy()

    # Group race results by (season, round) to avoid repeated full-table scans
    race_groups = {}
    for (season, rnd), group in race_results.groupby(["season", "round"]):
        race_groups[(int(season), int(rnd))] = group

    # Pre-build FastF1 team name -> constructor_id mapping for pit stop features
    # Maps (year, gp, fastf1_team) -> constructor_id
    _fastf1_team_to_constructor: dict[tuple[int, int, str], str] = {}
    if has_pit_stops and fastf1_laps is not None and not fastf1_laps.empty:
        race_laps_pit = fastf1_laps[fastf1_laps["session_type"] == "R"]
        if not race_laps_pit.empty and "Team" in race_laps_pit.columns:
            for (year, gp), group in race_laps_pit.groupby(["year", "gp"]):
                for driver_code, dlaps in group.groupby("Driver"):
                    driver_id = driver_code_map.get(driver_code)
                    if not driver_id:
                        continue
                    team = dlaps["Team"].iloc[0]
                    race_key = (int(year), int(gp))
                    rg = race_groups.get(race_key)
                    if rg is not None:
                        match = rg[rg["driver_id"] == driver_id]
                        if not match.empty:
                            cid = match.iloc[0].get("constructor_id")
                            if cid:
                                _fastf1_team_to_constructor[(int(year), int(gp), team)] = cid

    # Pre-index pit stop stats by (season, round) for O(1) race lookup
    _pit_stop_by_race: dict[tuple[int, int], list[tuple]] = {}
    for pit_key in pit_stop_stats:
        race_key = (pit_key[0], pit_key[1])
        _pit_stop_by_race.setdefault(race_key, []).append(pit_key)

    # Group qualifying by (season, round) and index by driver_id
    # Take the best position if duplicates exist (regular Q + sprint Q mixed)
    quali_index: dict[tuple[int, int], dict[str, int]] = {}
    if qualifying is not None and not qualifying.empty:
        for (season, rnd), group in qualifying.groupby(["season", "round"]):
            driver_pos = {}
            for _, row in group.iterrows():
                did = row["driver_id"]
                pos = int(row.get("position", 0))
                if did not in driver_pos or pos < driver_pos[did]:
                    driver_pos[did] = pos
            quali_index[(int(season), int(rnd))] = driver_pos

    # Get unique races in order
    unique_races = (
        race_results[["season", "round", "circuit_id"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
    )

    # Build regulation reset factors for ELO
    regulation_resets = {}
    for year, change in REGULATION_CHANGES.items():
        regulation_resets[year] = compute_elo_reset_factors(year)

    # Build ELO incrementally — we need the state BEFORE each race
    elo = F1EloSystem()
    feature_rows = []
    _prev_elo_season = None
    _reg_feature_cache: Dict[int, Dict[str, float]] = {}

    # Driver history accumulators (position-based + FastF1-based + sprint + track status)
    driver_history: dict[str, list[dict]] = {}
    driver_fastf1_history: dict[str, list[dict]] = {}
    driver_sprint_history: dict[str, list[dict]] = {}
    driver_track_status_history: dict[str, list[dict]] = {}
    # Constructor-level accumulator: (constructor_id, season) -> list of positions
    constructor_season_positions: dict[tuple[str, int], list[float]] = {}
    # Teammate qualifying H2H tracker: driver_id -> list of 1/0 (beat teammate or not) per season
    driver_quali_h2h: dict[str, list[int]] = {}
    # Career race counter: driver_id -> total races seen
    driver_career_races: dict[str, int] = {}
    # Constructor pace per circuit type: (constructor_id, season) -> {circuit_type: [positions]}
    constructor_pace_by_circuit_type: dict[tuple[str, int], dict[str, list[float]]] = {}
    # Team pit stop history: (team_name) -> list of {pit_time_avg, clean_pct}
    team_pit_stop_history: dict[str, list[dict]] = {}
    # Constructor-at-circuit history: (constructor_id, circuit_id) -> list of (season, position)
    constructor_circuit_history: dict[tuple[str, str], list[tuple[int, float]]] = {}
    # Driver qualifying-at-circuit history: (driver_id, circuit_id) -> list of quali positions
    driver_quali_circuit: dict[tuple[str, str], list[int]] = {}

    for _, race_info in unique_races.iterrows():
        season = int(race_info["season"])
        rnd = int(race_info["round"])
        circuit_id = race_info["circuit_id"]
        is_wet = (season, rnd) in wet_races

        # Apply regulation ELO reset at season boundaries
        if _prev_elo_season is not None and season != _prev_elo_season:
            if season in regulation_resets:
                reset = regulation_resets[season]
                elo.apply_regulation_reset(
                    season,
                    driver_retain=reset["driver"],
                    constructor_retain=reset["constructor"],
                )
        _prev_elo_season = season

        race_data = race_groups[(season, rnd)]

        # Get qualifying lookup for this race
        quali_lookup = quali_index.get((season, rnd), {})

        # Precompute field ELOs once per race
        ct = CIRCUIT_TYPES.get(circuit_id, DEFAULT_CIRCUIT_TYPE)
        field_elo_values = []
        for _, r in race_data.iterrows():
            rating = elo.overall.get(r["driver_id"])
            if rating:
                field_elo_values.append(rating.rating)
        field_elo_mean = np.mean(field_elo_values) if field_elo_values else 1500.0
        # Sort ascending for O(log n) rank via bisect
        field_elo_sorted_asc = sorted(field_elo_values)

        # Pre-compute championship leader points once per race (not per driver)
        _prev_key = None
        _leader_points = None
        if has_standings:
            if rnd > 1:
                _prev_key = (season, rnd - 1)
            else:
                prev_season_max = standings_max_round.get(season - 1)
                _prev_key = (season - 1, prev_season_max) if prev_season_max else None
            if _prev_key and _prev_key in driver_standings_index:
                _leader_points = max(
                    s["points"] for s in driver_standings_index[_prev_key].values()
                )

        for _, row in race_data.iterrows():
            driver_id = row["driver_id"]
            constructor_id = row.get("constructor_id", "unknown")

            # ── ELO Features (state BEFORE this race) ──
            features = {
                "season": season,
                "round": rnd,
                "circuit_id": circuit_id,
                "driver_id": driver_id,
                "constructor_id": constructor_id,
                "is_wet": int(is_wet),
            }

            d_elo = elo.overall.get(driver_id)
            features["elo_overall"] = d_elo.rating if d_elo else 1500.0
            features["elo_matches"] = d_elo.matches if d_elo else 0

            q_elo = elo.qualifying.get(driver_id)
            features["elo_qualifying"] = q_elo.rating if q_elo else 1500.0

            w_elo = elo.wet.get(driver_id)
            features["elo_wet"] = w_elo.rating if w_elo else 1500.0

            ct_elo = elo.by_circuit_type[ct].get(driver_id)
            features["elo_circuit_type"] = ct_elo.rating if ct_elo else 1500.0
            features["circuit_type"] = ct

            c_elo = elo.constructors.get(constructor_id)
            features["elo_constructor"] = c_elo.rating if c_elo else 1500.0

            # ── Regulation Features (season-level, cached) ──
            if season not in _reg_feature_cache:
                _reg_feature_cache[season] = build_regulation_features(season)
            features.update(_reg_feature_cache[season])

            # ELO rank via sorted list (O(n) per driver but O(n log n) sort once)
            features["elo_diff_vs_field"] = features["elo_overall"] - field_elo_mean
            my_elo = features["elo_overall"]
            features["elo_rank_in_field"] = len(field_elo_sorted_asc) - bisect.bisect_right(field_elo_sorted_asc, my_elo) + 1

            # ── Grid Position ──
            grid = int(row.get("grid", 0))
            features["grid"] = grid

            # Qualifying position (O(1) dict lookup instead of DataFrame scan)
            if quali_lookup:
                q_pos = quali_lookup.get(driver_id)
                if q_pos is not None:
                    features["quali_position"] = q_pos
                    # Grid penalty: positive = dropped positions (penalty applied)
                    if grid > 0:
                        features["grid_penalty"] = grid - q_pos
                        features["has_grid_penalty"] = 1 if grid > q_pos else 0

            # ── Qualifying Pace (Q1/Q2/Q3 times in seconds) ──
            quali_times = quali_times_index.get((season, rnd), {}).get(driver_id)
            if quali_times:
                best_q = None
                for qn in ["q3_s", "q2_s", "q1_s"]:
                    t = quali_times.get(qn)
                    if t is not None:
                        if best_q is None or t < best_q:
                            best_q = t

                if best_q is not None:
                    features["quali_best_time"] = best_q

                    # Pace vs field median (negative = faster than median)
                    median = quali_times.get("best_field_median")
                    if median is not None:
                        features["quali_delta_vs_field"] = best_q - median

                # Q1 to Q3 improvement (pressure performance)
                q1 = quali_times.get("q1_s")
                q3 = quali_times.get("q3_s")
                if q1 is not None and q3 is not None and q1 > 0:
                    features["quali_improvement"] = q1 - q3
                    features["quali_improvement_pct"] = (q1 - q3) / q1

                # Made it to Q3? Q2?
                features["quali_reached_q3"] = 1 if quali_times.get("q3_s") is not None else 0
                features["quali_reached_q2"] = 1 if quali_times.get("q2_s") is not None else 0

            # ── Sprint Race Features (2021+, same weekend predictor) ──
            sprint_data = sprint_index.get((season, rnd), {}).get(driver_id)
            if sprint_data:
                features["sprint_position"] = sprint_data["sprint_position"]
                features["sprint_grid"] = sprint_data["sprint_grid"]
                features["sprint_points"] = sprint_data["sprint_points"]
                # Grid vs sprint position delta (positive = gained positions)
                if sprint_data["sprint_position"] is not None and sprint_data["sprint_grid"] > 0:
                    features["sprint_grid_delta"] = sprint_data["sprint_grid"] - sprint_data["sprint_position"]
                features["sprint_dnf"] = sprint_data["sprint_dnf"]

            # ── Weather Conditions (from FastF1) ──
            race_weather = weather_index.get((season, rnd))
            if race_weather:
                for wk, wv in race_weather.items():
                    features[wk] = wv

            # ── Championship Standings (PREVIOUS round — no leakage) ──
            if has_standings:
                if _prev_key and _prev_key in driver_standings_index:
                    driver_standing = driver_standings_index[_prev_key].get(driver_id)
                    if driver_standing:
                        features["championship_position"] = driver_standing["position"]
                        features["championship_points"] = driver_standing["points"]
                        if _leader_points is not None:
                            features["points_to_leader"] = _leader_points - driver_standing["points"]
                        # Title contender: within mathematical contention
                        total_rounds_this_season = standings_max_round.get(season)
                        if total_rounds_this_season is None:
                            total_rounds_this_season = standings_max_round.get(season - 1, 22)
                        remaining_races = total_rounds_this_season - (rnd - 1)
                        features["title_contender"] = (
                            1 if features.get("points_to_leader", float("inf")) < remaining_races * 26 else 0
                        )

                    # Constructor championship position
                    if _prev_key in constructor_standings_index:
                        constructor_standing = constructor_standings_index[_prev_key].get(constructor_id)
                        if constructor_standing:
                            features["constructor_championship_pos"] = constructor_standing["position"]

            # ── Recent Form (rolling stats) ──
            history = driver_history.get(driver_id, [])
            if history:
                hist_df = pd.DataFrame(history)

                for w in _WINDOWS:
                    recent = hist_df.tail(w)
                    if not recent.empty:
                        features[f"pos_last{w}_mean"] = recent["position"].mean()
                        features[f"pos_last{w}_std"] = recent["position"].std()
                        features[f"pos_last{w}_best"] = recent["position"].min()
                        features[f"podium_rate_last{w}"] = (recent["position"] <= 3).mean()
                        features[f"points_rate_last{w}"] = (recent["position"] <= 10).mean()
                        features[f"dnf_rate_last{w}"] = recent["dnf"].mean()

                # Clean rolling averages (excluding compromised finishes)
                if "compromised" in hist_df.columns:
                    clean = hist_df[hist_df["compromised"] == 0]
                    if len(clean) >= 3:
                        clean_recent = clean.tail(5)
                        features["clean_pos_last5_mean"] = clean_recent["position"].mean()
                        features["clean_pos_last5_best"] = clean_recent["position"].min()
                    features["compromised_rate_last10"] = hist_df.tail(10)["compromised"].mean()

                # Circuit-specific history
                features.update(_DEFAULT_CIRCUIT)
                circuit_hist = hist_df[hist_df["circuit_id"] == circuit_id]
                if not circuit_hist.empty:
                    pos_col = circuit_hist["position"]
                    pos_values = pos_col.dropna().values
                    features["circuit_races"] = len(circuit_hist)
                    features["circuit_avg_pos"] = pos_col.mean()
                    features["circuit_best_pos"] = pos_col.min()
                    features["circuit_podium_rate"] = (pos_col <= 3).mean()

                    # Recency-weighted circuit performance
                    if len(pos_values) >= 2:
                        features["circuit_ewm_pos"] = _ewm(pos_values, 0.4)
                    elif len(pos_values) == 1:
                        features["circuit_ewm_pos"] = pos_values[0]

                    # Circuit-specific positions gained (grid → finish delta)
                    circuit_gains = circuit_hist[circuit_hist["grid"] > 0]
                    if not circuit_gains.empty:
                        deltas = circuit_gains["grid"] - circuit_gains["position"]
                        features["circuit_avg_gained"] = deltas.mean()
                        if len(deltas) >= 3:
                            features["circuit_recent_gained"] = deltas.tail(3).mean()

                    # Win streak at this circuit
                    wins = (pos_col == 1).values
                    streak = 0
                    for w in reversed(wins):
                        if w:
                            streak += 1
                        else:
                            break
                    if streak > 0:
                        features["circuit_win_streak"] = streak

                # Qualifying-at-circuit history (independent of race finishes)
                qc_hist = driver_quali_circuit.get((driver_id, circuit_id), [])
                if qc_hist:
                    features["circuit_quali_avg"] = np.mean(qc_hist)
                    features["circuit_quali_best"] = min(qc_hist)
                    if len(qc_hist) >= 2:
                        features["circuit_quali_recent"] = np.mean(qc_hist[-3:])

                # Season momentum
                season_hist = hist_df[hist_df["season"] == season]
                if not season_hist.empty:
                    features["season_races"] = len(season_hist)
                    features["season_avg_pos"] = season_hist["position"].mean()
                    features["season_points_total"] = season_hist["points"].sum()
                    if len(season_hist) >= 3:
                        features["season_trend"] = _linear_trend(
                            season_hist["position"].values[-3:]
                        )
                    else:
                        features["season_trend"] = 0.0
                else:
                    features.update(_DEFAULT_SEASON)
            else:
                features.update(_DEFAULT_ALL)

            # ── Sprint Rolling Features (from PREVIOUS sprints — no leakage) ──
            if has_sprints:
                sprint_hist = driver_sprint_history.get(driver_id, [])
                if sprint_hist:
                    recent_sp = sprint_hist[-5:]
                    features["sprint_avg_pos_last5"] = np.nanmean([h["sprint_position"] for h in recent_sp])
                    features["sprint_avg_delta_last5"] = np.mean([h["sprint_grid_delta"] for h in recent_sp])
                    features["sprint_dnf_rate"] = np.mean([h["sprint_dnf"] for h in recent_sp])

            # ── FastF1 Rolling Features (from PREVIOUS races — no leakage) ──
            if has_fastf1:
                f1_hist = driver_fastf1_history.get(driver_id, [])
                if f1_hist:
                    recent_f1 = f1_hist[-5:]
                    for col in _FASTF1_ROLLING_COLS:
                        vals = [h[col] for h in recent_f1 if col in h and h[col] is not None]
                        if vals:
                            features[f"f1_{col}_avg"] = np.mean(vals)

                    # Tire management trend (need more history)
                    deg_vals = [h["tire_deg_avg"] for h in f1_hist[-5:] if "tire_deg_avg" in h]
                    if len(deg_vals) >= 3:
                        features["f1_tire_management_trend"] = _linear_trend(np.array(deg_vals))

                    # Speed consistency
                    cons_vals = [h["lap_consistency"] for h in f1_hist[-5:] if "lap_consistency" in h]
                    if len(cons_vals) >= 3:
                        features["f1_pace_consistency"] = np.mean(cons_vals)

                # Track status rolling features (SC/VSC performance)
                ts_hist = driver_track_status_history.get(driver_id, [])
                if ts_hist:
                    recent_ts = ts_hist[-5:]
                    gains = [h["sc_pos_gain"] for h in recent_ts if "sc_pos_gain" in h]
                    if gains:
                        features["f1_sc_pos_gain_avg"] = np.mean(gains)
                    features["f1_sc_exposure"] = np.mean([h["sc_laps_pct"] for h in recent_ts])

            # ── Practice Session Pace (FP1/FP2/FP3 — same weekend) ──
            if has_practice:
                fp_data = practice_pace.get((season, rnd, driver_id))
                if fp_data:
                    features["fp_best_time"] = fp_data["fp_best_time"]
                    features["fp_total_laps"] = fp_data["fp_total_laps"]

                    # Practice pace vs field (pre-computed median)
                    fp_median = fp_field_medians.get((season, rnd))
                    if fp_median is not None:
                        features["fp_delta_vs_field"] = fp_data["fp_best_time"] - fp_median

                    # FP2 long run analysis (race pace indicator + tire degradation)
                    if "fp2_long_run_pace" in fp_data:
                        features["fp2_long_run_pace"] = fp_data["fp2_long_run_pace"]
                    if "fp2_deg_rate" in fp_data:
                        features["fp2_deg_rate"] = fp_data["fp2_deg_rate"]

            # ── Constructor Development Trajectory ──
            # O(1) lookup from pre-accumulated constructor positions
            c_key = (constructor_id, season)
            constructor_results = constructor_season_positions.get(c_key, [])
            if len(constructor_results) >= 4:
                features["constructor_season_trend"] = _linear_trend(
                    np.array(constructor_results[-10:])
                )
                features["constructor_season_avg"] = np.mean(constructor_results)

            # ── Constructor-at-Circuit Performance ──
            cc_key = (constructor_id, circuit_id)
            cc_hist = constructor_circuit_history.get(cc_key, [])
            if cc_hist:
                cc_positions = [p for _, p in cc_hist]
                features["constructor_circuit_avg"] = np.mean(cc_positions)
                features["constructor_circuit_races"] = len(cc_positions)
                if len(cc_positions) >= 2:
                    features["constructor_circuit_recent"] = np.mean(cc_positions[-3:])
                    # Trend: latest result vs baseline (last 3 seasons, excluding latest)
                    recent_cc = [p for s, p in cc_hist if s >= season - 3]
                    if len(recent_cc) >= 2:
                        features["constructor_circuit_trend"] = recent_cc[-1] - np.mean(recent_cc[:-1])

            # ── DNF Risk Indicators (list ops, no DataFrame overhead) ──
            # dnf_rate_last20 already computed in rolling stats above
            if history:
                dnf_streak = 0
                for h in reversed(history):
                    if h["dnf"] == 1:
                        dnf_streak += 1
                    else:
                        break
                features["dnf_streak"] = dnf_streak
                circuit_dnfs = [h["dnf"] for h in history if h["circuit_id"] == circuit_id]
                if len(circuit_dnfs) >= 2:
                    features["circuit_dnf_rate"] = np.mean(circuit_dnfs)

            # ── Teammate Comparison Features ──
            # Find teammate: same constructor_id in this race
            teammate_id = None
            for _, mate_row in race_data.iterrows():
                if (
                    mate_row["driver_id"] != driver_id
                    and mate_row.get("constructor_id") == constructor_id
                ):
                    teammate_id = mate_row["driver_id"]
                    break

            if teammate_id is not None:
                # ELO difference vs teammate
                mate_elo = elo.overall.get(teammate_id)
                my_elo_val = features["elo_overall"]
                mate_elo_val = mate_elo.rating if mate_elo else 1500.0
                features["teammate_elo_diff"] = my_elo_val - mate_elo_val

                # Qualifying H2H rate this season
                h2h_results = driver_quali_h2h.get(driver_id, [])
                if h2h_results:
                    features["h2h_quali_rate"] = np.mean(h2h_results)

                # Avg positions gained (grid - finish) over last 10 races
                if history:
                    recent10 = history[-10:]
                    gains = [
                        h["grid"] - h["position"]
                        for h in recent10
                        if h["grid"] > 0 and not np.isnan(h["position"])
                    ]
                    if gains:
                        features["avg_positions_gained"] = np.mean(gains)
                    # Last 5
                    recent5 = history[-5:]
                    gains5 = [
                        h["grid"] - h["position"]
                        for h in recent5
                        if h["grid"] > 0 and not np.isnan(h["position"])
                    ]
                    if gains5:
                        features["avg_positions_gained_last5"] = np.mean(gains5)

            # ── Momentum / Form Features ──
            if history and len(history) >= 2:
                # Exponentially weighted position average (alpha=0.3) over last 5
                recent5_pos = [
                    h["position"] for h in history[-5:]
                    if not np.isnan(h["position"])
                ]
                if recent5_pos:
                    features["momentum_score"] = _ewm(recent5_pos, 0.3)

                # Form vs season avg: current 3-race rolling avg minus season avg
                season_positions = [
                    h["position"] for h in history
                    if h["season"] == season and not np.isnan(h["position"])
                ]
                if len(season_positions) >= 3:
                    rolling3 = np.mean(season_positions[-3:])
                    season_avg = np.mean(season_positions)
                    features["form_vs_season_avg"] = rolling3 - season_avg

            # ── Circuit DNA Features ──
            dna = circuit_dna.get(circuit_id)
            dna_corr = None
            if dna:
                dna_corr = dna["grid_position_correlation"]
                features["circuit_grid_correlation"] = dna_corr
                features["circuit_avg_positions_changed"] = dna["avg_positions_changed"]
                features["circuit_front_row_win_rate"] = dna["front_row_win_rate"]
                features["circuit_front_row_lock_rate"] = dna["front_row_lock_rate"]
                features["circuit_attrition_rate"] = dna["attrition_rate"]
                features["circuit_position_variance"] = dna["position_variance"]

            # ── Grid × Circuit Interaction Features ──
            if grid > 0:
                is_street = 1 if circuit_id in _STREET_CIRCUITS else 0
                is_high_speed = 1 if circuit_id in _HIGH_SPEED_CIRCUITS else 0
                is_technical = 1 if circuit_id in _TECHNICAL_CIRCUITS else 0
                features["grid_x_street"] = grid * is_street
                features["grid_x_high_speed"] = grid * is_high_speed
                features["grid_x_technical"] = grid * is_technical

                # Grid weighted by how much it matters at this circuit
                if dna_corr is not None and not np.isnan(dna_corr):
                    features["grid_weighted_by_correlation"] = grid * dna_corr
                    features["grid_importance_score"] = dna_corr
                    features["grid_expected_finish"] = grid * dna_corr + (1 - dna_corr) * 10.0
                    features["front_row_at_this_circuit"] = (
                        dna["front_row_win_rate"] if grid <= 2 else 0.0
                    )

            # ── Overtaking Difficulty (circuit-level historical stats) ──
            ot_stats = circuit_overtaking.get(circuit_id)
            if ot_stats:
                features["circuit_overtaking_rate"] = ot_stats["avg_positions_gained"]
                # Expected grid delta: historical avg finish from this grid slot
                if grid > 0:
                    expected_finish = ot_stats["grid_finish_map"].get(grid)
                    if expected_finish is not None:
                        features["expected_grid_delta"] = grid - expected_finish

            # ── Constructor Upgrade Detection ──
            c_key_ct = (constructor_id, season)
            c_results = constructor_season_positions.get(c_key_ct, [])
            if len(c_results) >= 4:
                season_avg_c = np.mean(c_results)
                last2_avg = np.mean(c_results[-2:])
                # Pace jump: sudden improvement > 0.3 positions vs season average
                pace_jump_val = season_avg_c - last2_avg  # positive = improved
                features["constructor_pace_jump"] = 1.0 if pace_jump_val > 0.3 else 0.0
                features["constructor_pace_jump_magnitude"] = pace_jump_val

            # Pace vs previous similar circuit type
            ct_history = constructor_pace_by_circuit_type.get(c_key_ct, {})
            ct_positions = ct_history.get(ct, [])
            if ct_positions:
                features["constructor_pace_vs_prev_similar"] = np.mean(ct_positions)

            # ── Cold Start / Rookie Handling ──
            career_count = driver_career_races.get(driver_id, 0)
            features["career_races"] = career_count
            features["is_rookie"] = 1 if career_count < 5 else 0

            # Teammate ELO as a prior for rookies (useful when driver has no history)
            if teammate_id is not None:
                mate_elo_obj = elo.overall.get(teammate_id)
                features["teammate_elo_overall"] = mate_elo_obj.rating if mate_elo_obj else 1500.0

            # ── Pit Stop Performance (team-level from FastF1 stint data) ──
            if has_pit_stops:
                team_name = row.get("constructor_id", "unknown")
                # Use team pit stop history (last 5 races, from previous races)
                pit_hist = team_pit_stop_history.get(team_name, [])
                if pit_hist:
                    recent_pits = pit_hist[-5:]
                    features["avg_pit_stop_time"] = np.mean(
                        [h["pit_time_avg"] for h in recent_pits]
                    )
                    features["pit_stop_reliability"] = np.mean(
                        [h["clean_pct"] for h in recent_pits]
                    )

            # ── Target Variable ──
            features["position"] = row.get("position")
            features["points"] = row.get("points", 0)
            status = row.get("status", "")
            features["status"] = status
            features["dnf"] = 0 if _is_classified(status) else 1

            feature_rows.append(features)

            # Detect compromised finish: classified but lost far more positions
            # than typical for this circuit (incident/penalty during race)
            finish_pos = features["position"]
            compromised = 0
            if (
                features["dnf"] == 0
                and finish_pos is not None
                and not np.isnan(finish_pos)
                and grid > 0
                and dna
            ):
                positions_lost = finish_pos - grid
                avg_delta = dna["avg_positions_changed"]
                pos_std = dna["position_variance"]
                # Z-score: how many SDs beyond the circuit's average position change
                if pos_std > 0:
                    z = (positions_lost - avg_delta) / pos_std
                    compromised = 1 if z > 2.0 else 0
                elif positions_lost > avg_delta + 8:
                    compromised = 1

            # Update driver history (after feature extraction — no leakage)
            driver_history.setdefault(driver_id, []).append({
                "season": season,
                "round": rnd,
                "circuit_id": circuit_id,
                "constructor_id": constructor_id,
                "position": finish_pos if finish_pos is not None else np.nan,
                "grid": grid,
                "points": row.get("points", 0),
                "dnf": features["dnf"],
                "compromised": compromised,
            })

            # Update constructor season accumulator
            pos = row.get("position")
            if pos is not None and not np.isnan(pos):
                constructor_season_positions.setdefault(
                    (constructor_id, season), []
                ).append(pos)
                # Update constructor-at-circuit accumulator
                constructor_circuit_history.setdefault(
                    (constructor_id, circuit_id), []
                ).append((season, pos))

            # Update FastF1 history (after feature extraction — no leakage)
            f1_key = (season, rnd, driver_id)
            if f1_key in fastf1_stats:
                driver_fastf1_history.setdefault(driver_id, []).append(
                    fastf1_stats[f1_key]
                )

            # Update track status history
            if f1_key in track_status_stats:
                driver_track_status_history.setdefault(driver_id, []).append(
                    track_status_stats[f1_key]
                )

            # Update sprint history
            sprint_result = sprint_index.get((season, rnd), {}).get(driver_id)
            if sprint_result and sprint_result["sprint_position"] is not None:
                driver_sprint_history.setdefault(driver_id, []).append({
                    "sprint_position": sprint_result["sprint_position"],
                    "sprint_dnf": sprint_result["sprint_dnf"],
                    "sprint_grid_delta": (
                        sprint_result["sprint_grid"] - sprint_result["sprint_position"]
                        if sprint_result["sprint_grid"] > 0
                        else 0
                    ),
                })

            # Update qualifying-at-circuit + H2H trackers
            if quali_lookup:
                my_quali = quali_lookup.get(driver_id)
                if my_quali is not None:
                    driver_quali_circuit.setdefault(
                        (driver_id, circuit_id), []
                    ).append(my_quali)
                    if teammate_id is not None:
                        mate_quali = quali_lookup.get(teammate_id)
                        if mate_quali is not None:
                            driver_quali_h2h.setdefault(driver_id, []).append(
                                1 if my_quali < mate_quali else 0
                            )

            # Update career race counter (after feature extraction — no leakage)
            driver_career_races[driver_id] = driver_career_races.get(driver_id, 0) + 1

            # Update constructor pace by circuit type (after feature extraction)
            if pos is not None and not np.isnan(pos):
                constructor_pace_by_circuit_type.setdefault(
                    (constructor_id, season), {}
                ).setdefault(ct, []).append(pos)

        # Update team pit stop history once per race (after all drivers processed)
        if has_pit_stops:
            race_pit_keys = _pit_stop_by_race.get((season, rnd), [])
            for pit_key in race_pit_keys:
                fastf1_team = pit_key[2]
                cid = _fastf1_team_to_constructor.get((season, rnd, fastf1_team))
                if cid:
                    team_pit_stop_history.setdefault(cid, []).append(
                        pit_stop_stats[pit_key]
                    )

        # Update ELO AFTER extracting features (no data leakage)
        elo.process_race(season, rnd, circuit_id, race_data, is_wet)
        if quali_lookup:
            quali_df = qualifying[
                (qualifying["season"] == season) & (qualifying["round"] == rnd)
            ]
            elo.process_qualifying(quali_df)

    df = pd.DataFrame(feature_rows)

    # Log feature counts
    base_cols = [c for c in df.columns if not c.startswith("f1_")]
    f1_cols = [c for c in df.columns if c.startswith("f1_")]
    logger.info(
        f"Feature matrix: {len(df)} rows × {len(df.columns)} columns "
        f"({len(base_cols)} base + {len(f1_cols)} FastF1)"
    )

    return df


def prepare_training_data(
    feature_matrix: pd.DataFrame,
    target: str = "position",
    min_season: int = 1980,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for model training.

    Args:
        feature_matrix: Output of build_feature_matrix()
        target: "position" for regression, or create binary targets
        min_season: Skip very early seasons with less data

    Returns:
        (X, y) tuple ready for sklearn/xgboost
    """
    df = feature_matrix[feature_matrix["season"] >= min_season].copy()
    df = df.dropna(subset=[target])

    # Drop non-feature columns
    drop_cols = [
        "driver_id", "constructor_id", "circuit_id",
        "status", "position", "points", "dnf",
        "circuit_type",  # will encode
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Encode circuit type
    if "circuit_type" in df.columns:
        ct_dummies = pd.get_dummies(df["circuit_type"], prefix="ct")
        df = pd.concat([df, ct_dummies], axis=1)

    y = df[target].astype(float)
    X = df.drop(columns=drop_cols, errors="ignore")

    # Drop remaining non-numeric
    X = X.select_dtypes(include=[np.number])

    logger.info(f"Training data: {len(X)} samples, {len(X.columns)} features")

    return X, y
