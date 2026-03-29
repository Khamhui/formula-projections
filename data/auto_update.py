"""
Auto-update pipeline — keeps model current and predictions fresh.

Detects new race results, invalidates stale cache, re-ingests the current
season, retrains models, and predicts the next upcoming race.

Usage:
    python -m data.auto_update              # Check + update if needed
    python -m data.auto_update --force      # Force full cycle regardless
    python -m data.auto_update --dry-run    # Show what would happen
    python -m data.auto_update --predict-only  # Skip retrain, just predict next race

Designed to run via launchd/cron (e.g., every Monday 6 AM).
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "cache" / "processed"
JOLPICA_CACHE = Path(__file__).parent / "cache" / "jolpica"
STATE_FILE = Path(__file__).parent / "cache" / ".auto_update_state.json"
LOG_FILE = Path(__file__).parent / "cache" / "auto_update.log"


def _load_state() -> dict:
    """Load persistent state from last run."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def _save_state(state: dict):
    """Persist state for next run."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _get_schedule() -> pd.DataFrame:
    """Get the current season's F1 schedule from FastF1."""
    import fastf1
    from data.ingest.fastf1_ingest import setup_fastf1_cache

    setup_fastf1_cache()
    season = datetime.now().year
    return fastf1.get_event_schedule(season, include_testing=False)


def check_for_new_results() -> dict | None:
    """
    Check if a race has completed since our last update.

    Returns race info dict if update needed, None otherwise.
    """
    state = _load_state()
    last_update = state.get("last_race_ingested")  # "2026-R03" format
    now = datetime.now()

    schedule = _get_schedule()
    season = now.year

    # Find the most recent completed race (event date in the past)
    completed = []
    for _, event in schedule.iterrows():
        rnd = event.get("RoundNumber", 0)
        if rnd == 0:
            continue
        event_date = pd.to_datetime(event["EventDate"])
        # Race is complete if event date + 1 day has passed (results available ~2h after race)
        if event_date.date() < now.date():
            completed.append({
                "season": season,
                "round": int(rnd),
                "name": event["EventName"],
                "date": event_date.strftime("%Y-%m-%d"),
                "key": f"{season}-R{int(rnd):02d}",
            })

    if not completed:
        logger.info("No completed races found for %d", season)
        return None

    latest = completed[-1]

    if last_update and latest["key"] <= last_update:
        logger.info(
            "Already up to date (last ingested: %s, latest race: %s %s)",
            last_update, latest["key"], latest["name"],
        )
        return None

    logger.info(
        "New race detected: %s %s (last ingested: %s)",
        latest["key"], latest["name"], last_update or "never",
    )
    return latest


def find_next_race() -> dict | None:
    """Find the next upcoming race to predict."""
    now = datetime.now()
    schedule = _get_schedule()
    season = now.year

    for _, event in schedule.iterrows():
        rnd = event.get("RoundNumber", 0)
        if rnd == 0:
            continue
        event_date = pd.to_datetime(event["EventDate"])
        if event_date.date() >= now.date():
            return {
                "season": season,
                "round": int(rnd),
                "name": event["EventName"],
                "date": event_date.strftime("%Y-%m-%d"),
            }

    logger.info("No upcoming races for %d — season complete", season)
    return None


def invalidate_season_cache(season: int):
    """
    Remove Jolpica cache files for a specific season so fresh data is fetched.

    Cache files follow the pattern: {season}_*.json
    """
    if not JOLPICA_CACHE.exists():
        logger.debug("No Jolpica cache directory — nothing to invalidate")
        return

    count = 0
    for cache_file in JOLPICA_CACHE.glob(f"{season}_*.json"):
        cache_file.unlink()
        count += 1

    # Also invalidate season-specific endpoints with different naming
    for cache_file in JOLPICA_CACHE.glob("*.json"):
        if f"_{season}_" in cache_file.name or cache_file.name.startswith(f"{season}_"):
            cache_file.unlink()
            count += 1

    logger.info("Invalidated %d cache files for season %d", count, season)


def run_update(
    force: bool = False,
    dry_run: bool = False,
    predict_only: bool = False,
):
    """
    Main auto-update cycle:
    1. Check if a new race happened since last update
    2. Invalidate stale Jolpica cache for current season
    3. Re-ingest current season
    4. Rebuild features + retrain model
    5. Predict next upcoming race
    """
    start_time = time.time()
    now = datetime.now()
    season = now.year

    print(f"\n{'='*60}")
    print(f"  F1 Auto-Update — {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    # Step 1: Check for new results
    if predict_only:
        new_race = None
        print("  [skip] Predict-only mode — skipping ingest/retrain")
    elif force:
        new_race = {"key": "forced", "name": "forced update", "season": season}
        print("  [force] Forced update — running full cycle")
    else:
        print("Step 1: Checking for new race results...")
        new_race = check_for_new_results()
        if new_race is None:
            print("  No new results. Checking if prediction exists for next race...")
            next_race = find_next_race()
            if next_race:
                pred_path = DATA_DIR / f"prediction_{next_race['season']}_R{next_race['round']:02d}.csv"
                if pred_path.exists():
                    print(f"  Prediction for {next_race['name']} already exists. Nothing to do.")
                    return
                else:
                    print(f"  Missing prediction for {next_race['name']} — generating...")
                    predict_only = True
            else:
                print("  Season complete. Nothing to do.")
                return
        else:
            print(f"  New race: {new_race['name']} ({new_race['key']})")

    if dry_run:
        next_race = find_next_race()
        print(f"\n  [dry-run] Would: invalidate cache → ingest {season} → retrain → predict {next_race['name'] if next_race else 'N/A'}")
        return

    if not predict_only:
        # Step 2: Invalidate stale cache
        print(f"\nStep 2: Invalidating Jolpica cache for {season}...")
        invalidate_season_cache(season)

        # Step 3: Re-ingest current season (fast — only 1 year, merged with history)
        print(f"\nStep 3: Re-ingesting {season} season data...")
        from data.pipeline import step_ingest
        rr_path = DATA_DIR / "race_results.parquet"
        pre_rows = len(pd.read_parquet(rr_path)) if rr_path.exists() else 0
        step_ingest(start_year=season, end_year=season, merge=True)
        post_rows = len(pd.read_parquet(rr_path)) if rr_path.exists() else 0
        if post_rows < pre_rows * 0.5 and pre_rows > 100:
            raise RuntimeError(
                f"Data integrity check failed: race_results dropped from "
                f"{pre_rows} to {post_rows} rows after merge. Aborting."
            )
        print(f"  Data integrity OK: {post_rows} rows (was {pre_rows})")

        # Step 4: Rebuild features + retrain
        print("\nStep 4: Rebuilding feature matrix...")
        from data.pipeline import step_features, step_train
        step_features()

        print("\nStep 5: Retraining models...")
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, metrics = step_train()
            print(f"  Position MAE: {metrics.get('position_mae', 'N/A')}")
            print(f"  Podium accuracy: {metrics.get('podium_accuracy', 'N/A')}")
            print(f"  Winner accuracy: {metrics.get('winner_accuracy', 'N/A')}")
        except Exception as e:
            print(f"  Retrain failed ({e}), using existing model")
            metrics = {}

        # Update state
        if new_race and new_race.get("key") != "forced":
            state = _load_state()
            state["last_race_ingested"] = new_race["key"]
            state["last_update"] = now.isoformat()
            state["metrics"] = {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in metrics.items()
            }
            _save_state(state)

    # Step 6: Predict next race
    next_race = find_next_race()
    if next_race:
        print(f"\nStep 6: Predicting {next_race['name']} ({next_race['season']} R{next_race['round']})...")
        from data.predict_weekend import run_weekend_prediction
        run_weekend_prediction(
            season=next_race["season"],
            race_round=next_race["round"],
            n_simulations=10000,
        )
    else:
        print("\nNo upcoming race to predict — season complete.")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Auto-update complete in {elapsed:.0f}s")
    print(f"{'='*60}\n")

    # Update state with prediction info
    state = _load_state()
    state["last_prediction"] = {
        "race": next_race["name"] if next_race else None,
        "round": next_race["round"] if next_race else None,
        "timestamp": now.isoformat(),
    }
    state["last_run"] = now.isoformat()
    state["elapsed_seconds"] = round(elapsed)
    _save_state(state)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 Auto-Update Pipeline")
    parser.add_argument("--force", action="store_true", help="Force full update regardless of state")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without doing it")
    parser.add_argument("--predict-only", action="store_true", help="Skip ingest/retrain, just predict next race")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Suppress noisy FastF1 logs
    for noisy in ["fastf1", "req", "core", "fastf1.core", "fastf1.req",
                   "fastf1._api", "fastf1.logger"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Log to both console and file
    handlers = [logging.StreamHandler()]
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(LOG_FILE))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    run_update(
        force=args.force,
        dry_run=args.dry_run,
        predict_only=args.predict_only,
    )
