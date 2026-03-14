"""
OpenF1 penalty extractor — parses race control messages for grid/time penalties.

Extracts pre-race grid penalties and in-race time penalties from
OpenF1's race_control endpoint (2023+).
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from data.ingest.openf1_client import OpenF1Client

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "cache" / "processed"

# Regex patterns for penalty extraction from race control messages
_RE_GRID_PENALTY = re.compile(
    r'(\d+)\s*(?:GRID\s*)?PLACE\s+GRID\s+PENALTY',
    re.IGNORECASE,
)
_RE_PIT_LANE_START = re.compile(
    r'PIT\s*LANE\s+START',
    re.IGNORECASE,
)
_RE_TIME_PENALTY = re.compile(
    r'(\d+)\s+SECOND\s+TIME\s+PENALTY',
    re.IGNORECASE,
)
_RE_BACK_OF_GRID = re.compile(
    r'BACK\s+OF\s+(?:THE\s+)?GRID',
    re.IGNORECASE,
)
_RE_CAR_NUMBER = re.compile(r'CAR\s+(\d+)')
_RE_DRIVER_CODE = re.compile(r'\(([A-Z]{3})\)')


def extract_penalties_from_session(
    client: OpenF1Client,
    session_key: int,
    session_type: str = "race",
) -> list[dict]:
    """
    Extract penalty info from race control messages for a session.

    Returns list of dicts with:
        driver_number, driver_code, penalty_type, penalty_value, message
    """
    try:
        rc = client.get_race_control(session_key)
    except Exception as e:
        logger.warning(f"Failed to get race control for session {session_key}: {e}")
        return []

    if rc.empty:
        return []

    penalties = []
    for _, msg in rc.iterrows():
        text = str(msg.get("message", ""))
        if not text:
            continue

        driver_number = msg.get("driver_number")
        # Try extracting from message text if not in structured field
        if pd.isna(driver_number):
            car_match = _RE_CAR_NUMBER.search(text)
            if car_match:
                driver_number = int(car_match.group(1))

        code_match = _RE_DRIVER_CODE.search(text)
        driver_code = code_match.group(1) if code_match else None

        penalty = None

        # Grid penalties (pre-race)
        grid_match = _RE_GRID_PENALTY.search(text)
        if grid_match:
            penalty = {
                "penalty_type": "grid_drop",
                "penalty_value": int(grid_match.group(1)),
            }

        back_match = _RE_BACK_OF_GRID.search(text)
        if back_match:
            penalty = {
                "penalty_type": "back_of_grid",
                "penalty_value": 20,  # effective max grid drop
            }

        pit_match = _RE_PIT_LANE_START.search(text)
        if pit_match:
            penalty = {
                "penalty_type": "pit_lane_start",
                "penalty_value": 20,
            }

        # Time penalties (in-race)
        time_match = _RE_TIME_PENALTY.search(text)
        if time_match:
            penalty = {
                "penalty_type": "time_penalty",
                "penalty_value": int(time_match.group(1)),
            }

        if penalty:
            penalty["driver_number"] = driver_number
            penalty["driver_code"] = driver_code
            penalty["session_key"] = session_key
            penalty["session_type"] = session_type
            penalty["message"] = text
            penalties.append(penalty)

    return penalties


def ingest_penalties(
    start_year: int = 2023,
    end_year: int = 2026,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Ingest all penalty data from OpenF1 for the given year range.
    Saves to parquet and returns DataFrame.
    """
    client = OpenF1Client()
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    all_penalties = []

    for year in range(start_year, end_year + 1):
        logger.info(f"Ingesting penalties for {year}...")

        try:
            sessions = client.get_sessions(year=year)
        except Exception as e:
            logger.warning(f"Failed to get sessions for {year}: {e}")
            continue

        if sessions.empty:
            continue

        for _, sess in sessions.iterrows():
            sk = sess["session_key"]
            st = str(sess.get("session_type", "")).lower()
            if st not in ("race", "qualifying", "sprint"):
                continue

            penalties = extract_penalties_from_session(client, sk, st)
            for p in penalties:
                p["year"] = year
                p["meeting_key"] = sess.get("meeting_key")
                p["session_name"] = sess.get("session_name")

            if penalties:
                all_penalties.extend(penalties)
                logger.info(f"  {sess.get('session_name', sk)}: {len(penalties)} penalties")

    if not all_penalties:
        logger.info("No penalties found")
        return pd.DataFrame()

    df = pd.DataFrame(all_penalties)
    df.to_parquet(out / "penalties.parquet", index=False)
    logger.info(f"Saved {len(df)} penalties to {out / 'penalties.parquet'}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = ingest_penalties(start_year=2024, end_year=2025)
    if not df.empty:
        print(f"\nPenalties: {len(df)} total")
        print(df.groupby("penalty_type").size())
        print(f"\nSample:")
        print(df[["year", "session_name", "driver_code", "penalty_type", "penalty_value"]].head(20))
