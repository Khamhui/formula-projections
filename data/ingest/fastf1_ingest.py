"""
FastF1 ingestion — granular lap times, telemetry, tire data, weather (2018+).
Enriches historical Jolpica data with detailed session-level stats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache" / "fastf1"
OUTPUT_DIR = Path(__file__).parent.parent / "cache" / "processed"


def setup_fastf1_cache():
    """Configure FastF1's built-in caching."""
    import fastf1
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def extract_session_laps(year: int, gp: Union[str, int], session_type: str = "R") -> Optional[pd.DataFrame]:
    """
    Extract lap data from a session.

    Args:
        year: Season year (2018+)
        gp: Grand Prix name or round number
        session_type: "R" (race), "Q" (qualifying), "S" (sprint),
                      "FP1", "FP2", "FP3" (practice)
    """
    import fastf1

    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=False, weather=True, messages=False)
        laps = session.laps
    except Exception as e:
        logger.warning(f"Failed to load {year} {gp} {session_type}: {e}")
        return None
    if laps.empty:
        return None

    # Convert timedeltas to seconds for easier processing
    time_cols = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
    df = laps.copy()
    for col in time_cols:
        if col in df.columns:
            df[f"{col}_s"] = df[col].dt.total_seconds()

    # Add session metadata
    df["year"] = year
    df["gp"] = str(gp)
    df["session_type"] = session_type
    df["event_name"] = session.event["EventName"]
    df["circuit_key"] = session.event.get("CircuitKey", "")

    # Extract weather at session level
    weather = session.weather_data
    if weather is not None and not weather.empty:
        df["air_temp_avg"] = weather["AirTemp"].mean()
        df["track_temp_avg"] = weather["TrackTemp"].mean()
        df["humidity_avg"] = weather["Humidity"].mean()
        df["rainfall"] = weather["Rainfall"].any()
        df["wind_speed_avg"] = weather["WindSpeed"].mean()

    # Select key columns
    keep_cols = [
        "year", "gp", "session_type", "event_name",
        "Driver", "DriverNumber", "Team",
        "LapNumber", "LapTime_s", "Stint",
        "Sector1Time_s", "Sector2Time_s", "Sector3Time_s",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "Compound", "TyreLife", "FreshTyre",
        "Position", "IsPersonalBest", "IsAccurate",
        "TrackStatus", "Deleted", "DeletedReason",
        "air_temp_avg", "track_temp_avg", "humidity_avg",
        "rainfall", "wind_speed_avg",
    ]

    available = [c for c in keep_cols if c in df.columns]
    return df[available].reset_index(drop=True)


def extract_telemetry_summary(year: int, gp: Union[str, int], session_type: str = "R") -> Optional[pd.DataFrame]:
    """
    Extract aggregated telemetry stats per driver per lap.
    Not the raw 3.7Hz data — summarized for ML features.
    """
    import fastf1

    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=True, weather=False, messages=False)
    except Exception as e:
        logger.warning(f"Failed to load telemetry {year} {gp} {session_type}: {e}")
        return None

    rows = []
    for _, lap in session.laps.iterrows():
        try:
            tel = lap.get_telemetry()
            if tel.empty:
                continue

            rows.append({
                "year": year,
                "gp": str(gp),
                "session_type": session_type,
                "driver": lap["Driver"],
                "lap_number": lap["LapNumber"],
                "avg_speed": tel["Speed"].mean(),
                "max_speed": tel["Speed"].max(),
                "avg_throttle": tel["Throttle"].mean(),
                "full_throttle_pct": (tel["Throttle"] >= 95).mean(),
                "braking_pct": tel["Brake"].mean(),
                "avg_rpm": tel["RPM"].mean(),
                "gear_changes": (tel["nGear"].diff() != 0).sum(),
                "drs_usage_pct": (tel["DRS"] >= 10).mean(),
            })
        except Exception:
            continue

    return pd.DataFrame(rows) if rows else None


def ingest_season_laps(year: int, session_types: list[str] = None) -> pd.DataFrame:
    """Ingest all lap data for an entire season. Call setup_fastf1_cache() before this."""
    import fastf1

    session_types = session_types or ["R", "Q"]

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    all_laps = []
    consecutive_failures = 0

    for _, event in tqdm(schedule.iterrows(), total=len(schedule), desc=f"FastF1 {year}"):
        round_num = event.get("RoundNumber", 0)
        if round_num == 0:
            continue

        for st in session_types:
            laps = extract_session_laps(year, round_num, st)
            if laps is not None:
                all_laps.append(laps)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 6:
                    logger.warning(f"6 consecutive failures — likely rate limited. Stopping {year}.")
                    break
        else:
            continue
        break

    if not all_laps:
        return pd.DataFrame()

    return pd.concat(all_laps, ignore_index=True)


def ingest_all(
    start_year: int = 2018,
    end_year: int = 2025,
    output_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Full FastF1 ingestion pipeline.
    Downloads lap data for all seasons and saves to parquet.
    """
    setup_fastf1_cache()
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    all_laps = []

    for year in range(start_year, end_year + 1):
        logger.info(f"Ingesting FastF1 data for {year}...")

        laps = ingest_season_laps(year, session_types=["R", "Q", "FP1", "FP2", "FP3"])
        if not laps.empty:
            all_laps.append(laps)
            logger.info(f"  {year}: {len(laps)} laps")

    datasets = {}

    if all_laps:
        df = pd.concat(all_laps, ignore_index=True)
        df.to_parquet(out / "fastf1_laps.parquet", index=False)
        datasets["laps"] = df
        logger.info(f"Total laps: {len(df)}")

    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets = ingest_all(start_year=2018, end_year=2025)
    for name, df in datasets.items():
        print(f"  {name}: {len(df)} rows, {len(df.columns)} columns")
