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
"""

import bisect
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from data.features.elo import F1EloSystem, CIRCUIT_TYPES, DEFAULT_CIRCUIT_TYPE

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


def _linear_trend(values: np.ndarray) -> float:
    """Slope of a linear fit over a sequence of values."""
    return np.polyfit(range(len(values)), values, 1)[0]


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


def _build_standings_index(
    data_dir: Path,
) -> tuple[dict, dict]:
    """
    Load championship standings parquet files and build lookup indices.

    Returns:
        driver_standings_index: dict[(season, round), dict[driver_id, {position, points, wins}]]
        constructor_standings_index: dict[(season, round), dict[constructor_id, {position, points}]]
    """
    driver_standings_index: dict[tuple[int, int], dict[str, dict]] = {}
    constructor_standings_index: dict[tuple[int, int], dict[str, dict]] = {}

    # Load driver standings
    driver_path = data_dir / "driver_standings.parquet"
    if driver_path.exists():
        df = pd.read_parquet(driver_path)
        for (season, rnd), group in df.groupby(["season", "round"]):
            drivers = {}
            for _, row in group.iterrows():
                drivers[row["driver_id"]] = {
                    "position": int(row.get("position", 0)),
                    "points": float(row.get("points", 0)),
                    "wins": int(row.get("wins", 0)),
                }
            driver_standings_index[(int(season), int(rnd))] = drivers
        logger.info(f"Loaded driver standings for {len(driver_standings_index)} race rounds")
    else:
        logger.warning(f"Driver standings not found at {driver_path}")

    # Load constructor standings
    constructor_path = data_dir / "constructor_standings.parquet"
    if constructor_path.exists():
        df = pd.read_parquet(constructor_path)
        for (season, rnd), group in df.groupby(["season", "round"]):
            constructors = {}
            for _, row in group.iterrows():
                constructors[row["constructor_id"]] = {
                    "position": int(row.get("position", 0)),
                    "points": float(row.get("points", 0)),
                }
            constructor_standings_index[(int(season), int(rnd))] = constructors
        logger.info(f"Loaded constructor standings for {len(constructor_standings_index)} race rounds")
    else:
        logger.warning(f"Constructor standings not found at {constructor_path}")

    return driver_standings_index, constructor_standings_index


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

    # Pre-compute championship standings index
    driver_standings_index, constructor_standings_index = _build_standings_index(DATA_DIR)
    has_standings = bool(driver_standings_index)

    # Build max-round-per-season lookup for previous-season fallback
    standings_max_round: dict[int, int] = {}
    for (s, r) in driver_standings_index:
        if s not in standings_max_round or r > standings_max_round[s]:
            standings_max_round[s] = r

    # Sort chronologically
    race_results = race_results.sort_values(["season", "round", "position"]).copy()

    # Group race results by (season, round) to avoid repeated full-table scans
    race_groups = {}
    for (season, rnd), group in race_results.groupby(["season", "round"]):
        race_groups[(int(season), int(rnd))] = group

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

    # Build ELO incrementally — we need the state BEFORE each race
    elo = F1EloSystem()
    feature_rows = []

    # Driver history accumulators (position-based + FastF1-based + sprint + track status)
    driver_history: dict[str, list[dict]] = {}
    driver_fastf1_history: dict[str, list[dict]] = {}
    driver_sprint_history: dict[str, list[dict]] = {}
    driver_track_status_history: dict[str, list[dict]] = {}
    # Constructor-level accumulator: (constructor_id, season) -> list of positions
    constructor_season_positions: dict[tuple[str, int], list[float]] = {}
    # Teammate qualifying H2H tracker: driver_id -> list of 1/0 (beat teammate or not) per season
    driver_quali_h2h: dict[str, list[int]] = {}

    for _, race_info in unique_races.iterrows():
        season = int(race_info["season"])
        rnd = int(race_info["round"])
        circuit_id = race_info["circuit_id"]
        is_wet = (season, rnd) in wet_races

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
                # Use previous round's standings; for round 1, use final round of previous season
                if rnd > 1:
                    prev_key = (season, rnd - 1)
                else:
                    prev_season_max = standings_max_round.get(season - 1)
                    prev_key = (season - 1, prev_season_max) if prev_season_max else None

                if prev_key and prev_key in driver_standings_index:
                    driver_standing = driver_standings_index[prev_key].get(driver_id)
                    if driver_standing:
                        features["championship_position"] = driver_standing["position"]
                        features["championship_points"] = driver_standing["points"]
                        # Points gap to leader
                        leader_points = max(
                            s["points"] for s in driver_standings_index[prev_key].values()
                        )
                        features["points_to_leader"] = leader_points - driver_standing["points"]
                        # Title contender: within mathematical contention
                        # Rough estimate: remaining_races * 26 (max points per race)
                        total_rounds_this_season = standings_max_round.get(season)
                        if total_rounds_this_season is None:
                            # Estimate from previous season if current not yet complete
                            total_rounds_this_season = standings_max_round.get(season - 1, 22)
                        remaining_races = total_rounds_this_season - (rnd - 1)
                        features["title_contender"] = (
                            1 if features["points_to_leader"] < remaining_races * 26 else 0
                        )

                    # Constructor championship position
                    if prev_key in constructor_standings_index:
                        constructor_standing = constructor_standings_index[prev_key].get(constructor_id)
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

                # Circuit-specific history
                circuit_hist = hist_df[hist_df["circuit_id"] == circuit_id]
                if not circuit_hist.empty:
                    features["circuit_races"] = len(circuit_hist)
                    features["circuit_avg_pos"] = circuit_hist["position"].mean()
                    features["circuit_best_pos"] = circuit_hist["position"].min()
                    features["circuit_podium_rate"] = (circuit_hist["position"] <= 3).mean()
                else:
                    features.update(_DEFAULT_CIRCUIT)

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

            # ── DNF Risk Indicators (list ops, no DataFrame overhead) ──
            history = driver_history.get(driver_id, [])
            if history:
                recent20 = history[-20:]
                features["dnf_rate_last20"] = np.mean([h["dnf"] for h in recent20])
                # Recent DNF streak (consecutive DNFs at end)
                dnf_streak = 0
                for h in reversed(history):
                    if h["dnf"] == 1:
                        dnf_streak += 1
                    else:
                        break
                features["dnf_streak"] = dnf_streak
                # Circuit-specific DNF rate
                circuit_dnfs = [h["dnf"] for h in history if h["circuit_id"] == circuit_id]
                if len(circuit_dnfs) >= 2:
                    features["circuit_dnf_rate"] = np.mean(circuit_dnfs)

            # ── Teammate Comparison Features ──
            history = driver_history.get(driver_id, [])
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
                    alpha = 0.3
                    ewm = recent5_pos[0]
                    for p in recent5_pos[1:]:
                        ewm = alpha * p + (1 - alpha) * ewm
                    features["momentum_score"] = ewm

                # Form vs season avg: current 3-race rolling avg minus season avg
                season_positions = [
                    h["position"] for h in history
                    if h["season"] == season and not np.isnan(h["position"])
                ]
                if len(season_positions) >= 3:
                    rolling3 = np.mean(season_positions[-3:])
                    season_avg = np.mean(season_positions)
                    features["form_vs_season_avg"] = rolling3 - season_avg

            # ── Target Variable ──
            features["position"] = row.get("position")
            features["points"] = row.get("points", 0)
            status = row.get("status", "")
            features["status"] = status
            features["dnf"] = 0 if _is_classified(status) else 1

            feature_rows.append(features)

            # Update driver history (after feature extraction — no leakage)
            driver_history.setdefault(driver_id, []).append({
                "season": season,
                "round": rnd,
                "circuit_id": circuit_id,
                "constructor_id": constructor_id,
                "position": row.get("position", np.nan),
                "grid": row.get("grid", 0),
                "points": row.get("points", 0),
                "dnf": features["dnf"],
            })

            # Update constructor season accumulator
            pos = row.get("position")
            if pos is not None and not np.isnan(pos):
                constructor_season_positions.setdefault(
                    (constructor_id, season), []
                ).append(pos)

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

            # Update qualifying H2H tracker (after feature extraction — no leakage)
            if quali_lookup and teammate_id is not None:
                my_quali = quali_lookup.get(driver_id)
                mate_quali = quali_lookup.get(teammate_id)
                if my_quali is not None and mate_quali is not None:
                    driver_quali_h2h.setdefault(driver_id, []).append(
                        1 if my_quali < mate_quali else 0
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
