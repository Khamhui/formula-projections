"""
Throttled FastF1 historical data fetch.

Downloads lap data for 2018–2025, one season at a time with delays
to avoid rate limiting. Merges into existing fastf1_laps.parquet.

Usage:
    python -m data.ingest.fetch_fastf1_history
    python -m data.ingest.fetch_fastf1_history --start 2022 --end 2025
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "cache" / "processed"
LAPS_PATH = OUTPUT_DIR / "fastf1_laps.parquet"


def fetch_season(year: int, session_types: list[str] = None) -> pd.DataFrame:
    """Fetch one season with per-round throttling to avoid rate limits."""
    import fastf1
    from data.ingest.fastf1_ingest import extract_session_laps, setup_fastf1_cache

    setup_fastf1_cache()
    session_types = session_types or ["R", "Q", "FP1", "FP2", "FP3"]

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        logger.error("Failed to load %d schedule: %s", year, e)
        return pd.DataFrame()

    all_laps = []
    consecutive_failures = 0

    for _, event in schedule.iterrows():
        round_num = event.get("RoundNumber", 0)
        if round_num == 0:
            continue

        round_laps = []
        for st in session_types:
            laps = extract_session_laps(year, round_num, st)
            if laps is not None:
                round_laps.append(laps)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 6:
                    logger.warning(
                        "6 consecutive failures at %d R%d — rate limited. "
                        "Saving progress and stopping.", year, round_num
                    )
                    if all_laps:
                        return pd.concat(all_laps, ignore_index=True)
                    return pd.DataFrame()

        if round_laps:
            all_laps.extend(round_laps)
            total = sum(len(l) for l in round_laps)
            logger.info("  %d R%02d: %d laps", year, round_num, total)

        # Throttle between rounds to avoid rate limits
        time.sleep(1.5)

    if not all_laps:
        return pd.DataFrame()
    return pd.concat(all_laps, ignore_index=True)


def merge_into_existing(new_laps: pd.DataFrame, year: int):
    """Merge new laps for a season into the existing parquet file."""
    if new_laps.empty:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if LAPS_PATH.exists():
        existing = pd.read_parquet(LAPS_PATH)
        keep = existing[existing["year"] != year]
        combined = pd.concat([keep, new_laps], ignore_index=True)
        combined = combined.sort_values(["year", "gp"]).reset_index(drop=True)
    else:
        combined = new_laps

    combined.to_parquet(LAPS_PATH, index=False)
    logger.info("Saved %d total laps to %s", len(combined), LAPS_PATH.name)


def fetch_all(start_year: int = 2018, end_year: int = 2025):
    """Fetch all seasons, merging each one as it completes."""
    # Check what we already have
    existing_years = set()
    if LAPS_PATH.exists():
        existing = pd.read_parquet(LAPS_PATH)
        existing_years = set(existing["year"].unique())
        logger.info("Existing data: %d laps for years %s", len(existing), sorted(existing_years))

    for year in range(start_year, end_year + 1):
        if year in existing_years:
            logger.info("Skipping %d — already fetched", year)
            continue

        print(f"\n{'='*60}")
        print(f"  Fetching FastF1 data for {year}")
        print(f"{'='*60}")

        laps = fetch_season(year)
        if not laps.empty:
            merge_into_existing(laps, year)
            print(f"  {year}: {len(laps)} laps saved")
        else:
            print(f"  {year}: no data available")

        # Longer pause between seasons
        if year < end_year:
            print("  Waiting 5s before next season...")
            time.sleep(5)

    # Final summary
    if LAPS_PATH.exists():
        final = pd.read_parquet(LAPS_PATH)
        years = sorted(final["year"].unique())
        print(f"\nDone: {len(final)} total laps across {len(years)} seasons ({years[0]}-{years[-1]})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch FastF1 historical lap data")
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    for noisy in ["fastf1", "req", "core", "fastf1.core", "fastf1.req",
                   "fastf1._api", "fastf1.logger"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    fetch_all(start_year=args.start, end_year=args.end)
