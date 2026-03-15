"""
Race Weekend Predictor — fetch live session data and predict the race.

Fetches FP1/Sprint Q/Sprint/Qualifying from FastF1, updates the feature matrix,
and runs prediction + Monte Carlo simulation for the upcoming race.

Usage:
    python -m data.predict_weekend                           # Auto-detect next race
    python -m data.predict_weekend --season 2026 --round 2   # Specific race
    python -m data.predict_weekend --refresh                 # Force re-fetch session data
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "cache" / "processed"
MODEL_DIR = Path(__file__).parent / "cache" / "models"

# FastF1 session types to fetch for a race weekend
WEEKEND_SESSIONS = ["FP1", "FP2", "FP3", "SQ", "S", "Q"]
# Sprint weekends don't have FP2/FP3 — FastF1 will return None, which we handle


def fetch_weekend_sessions(
    season: int,
    race_round: int,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch all available session data for a race weekend from FastF1.

    Returns DataFrame of laps across all completed sessions.
    """
    from data.ingest.fastf1_ingest import extract_session_laps, setup_fastf1_cache

    setup_fastf1_cache()

    all_laps = []
    for session_type in WEEKEND_SESSIONS:
        try:
            laps = extract_session_laps(season, race_round, session_type)
            if laps is not None and not laps.empty:
                logger.info(
                    f"  {session_type}: {len(laps)} laps, "
                    f"{laps['Driver'].nunique()} drivers"
                )
                all_laps.append(laps)
        except Exception as e:
            logger.debug(f"  {session_type}: not available ({e})")

    if not all_laps:
        logger.warning("No session data found for %d Round %d", season, race_round)
        return pd.DataFrame()

    return pd.concat(all_laps, ignore_index=True)


def update_fastf1_laps(weekend_laps: pd.DataFrame) -> pd.DataFrame:
    """Append weekend laps to the existing FastF1 laps parquet."""
    laps_path = DATA_DIR / "fastf1_laps.parquet"

    if laps_path.exists():
        existing = pd.read_parquet(laps_path)
        # Deduplicate: remove any existing data for this GP + year
        year = weekend_laps["year"].iloc[0]
        gp = weekend_laps["gp"].iloc[0]
        mask = ~((existing["year"] == year) & (existing["gp"] == gp))
        existing = existing[mask]
        combined = pd.concat([existing, weekend_laps], ignore_index=True)
    else:
        combined = weekend_laps

    combined.to_parquet(laps_path, index=False)
    logger.info(f"FastF1 laps updated: {len(combined)} total rows")
    return combined


def get_sprint_results(season: int, race_round: int) -> Optional[pd.DataFrame]:
    """Extract sprint race results from FastF1 session data."""
    from data.ingest.fastf1_ingest import setup_fastf1_cache
    import fastf1

    setup_fastf1_cache()

    try:
        session = fastf1.get_session(season, race_round, "S")
        session.load(telemetry=False, weather=False, messages=False)
    except Exception:
        return None

    if session.results.empty:
        return None

    results = session.results
    rows = []
    for _, r in results.iterrows():
        driver_code = r.get("Abbreviation", "")
        rows.append({
            "driver_code": driver_code,
            "position": r.get("Position"),
            "grid": r.get("GridPosition"),
            "points": r.get("Points", 0),
            "status": r.get("Status", ""),
        })

    return pd.DataFrame(rows)


def get_qualifying_results(season: int, race_round: int) -> Optional[pd.DataFrame]:
    """Extract qualifying results (grid order) from FastF1."""
    from data.ingest.fastf1_ingest import setup_fastf1_cache
    import fastf1

    setup_fastf1_cache()

    try:
        session = fastf1.get_session(season, race_round, "Q")
        session.load(telemetry=False, weather=False, messages=True)
    except Exception:
        return None

    if session.results.empty:
        return None

    results = session.results
    rows = []
    for _, r in results.iterrows():
        rows.append({
            "driver_code": r.get("Abbreviation", ""),
            "position": r.get("Position"),
            "q1": r.get("Q1"),
            "q2": r.get("Q2"),
            "q3": r.get("Q3"),
        })

    df = pd.DataFrame(rows)
    # Convert timedeltas to strings for compatibility
    for col in ["q1", "q2", "q3"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(x).split(" ")[-1] if pd.notna(x) else None
            )
    return df


def inject_upcoming_race(season: int, race_round: int) -> str:
    """
    Inject placeholder rows into race_results.parquet for an upcoming race.

    Uses qualifying data (grid positions) and previous race data (driver/constructor IDs)
    to create rows with NaN position (the prediction target).

    Returns the circuit_id for the race.
    """
    import fastf1
    from data.ingest.fastf1_ingest import setup_fastf1_cache

    setup_fastf1_cache()

    # Get circuit info from FastF1 schedule
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    event = schedule[schedule["RoundNumber"] == race_round].iloc[0]
    event_name = event["EventName"]

    # Map event name to circuit_id (match existing convention)
    circuit_id = _event_to_circuit_id(event_name)
    logger.info(f"Circuit: {circuit_id} ({event_name})")

    # Load existing data
    rr = pd.read_parquet(DATA_DIR / "race_results.parquet")

    # Check if this race already exists
    existing = rr[(rr["season"] == season) & (rr["round"] == race_round)]
    if not existing.empty:
        logger.info(f"Round {race_round} already in race_results (completed race)")
        return existing["circuit_id"].iloc[0]

    # Get qualifying to know the grid + drivers
    quali = pd.read_parquet(DATA_DIR / "qualifying.parquet")
    q_race = quali[(quali["season"] == season) & (quali["round"] == race_round)]

    def _placeholder(driver_id, driver_code, constructor_id, grid):
        return {
            "season": season, "round": race_round, "circuit_id": circuit_id,
            "driver_id": driver_id, "driver_code": driver_code,
            "constructor_id": constructor_id, "grid": grid,
            "position": None, "points": 0, "status": "Pending",
        }

    rows = []
    if not q_race.empty:
        for _, qrow in q_race.iterrows():
            rows.append(_placeholder(
                qrow["driver_id"], qrow.get("driver_code", ""),
                qrow.get("constructor_id", "unknown"), int(qrow.get("position", 0)),
            ))
    else:
        # No qualifying yet — estimate grid from latest race lineup
        logger.info("No qualifying data — estimating grid from latest race lineup")
        target_season = season if (rr["season"] == season).any() else rr["season"].max()
        latest = rr[rr["season"] == target_season]
        lineup = latest[latest["round"] == latest["round"].max()].sort_values("position")
        for grid_pos, (_, row) in enumerate(lineup.iterrows(), 1):
            rows.append(_placeholder(
                row["driver_id"], row.get("driver_code", ""),
                row.get("constructor_id", "unknown"), grid_pos,
            ))

    if not rows:
        logger.error("No driver data available for %d Round %d", season, race_round)
        return circuit_id

    placeholder = pd.DataFrame(rows)
    combined = pd.concat([rr, placeholder], ignore_index=True)
    combined.to_parquet(DATA_DIR / "race_results.parquet", index=False)
    logger.info(
        f"Injected {len(rows)} placeholder rows for {season} Round {race_round}"
    )

    return circuit_id


def _event_to_circuit_id(event_name: str) -> str:
    """Convert FastF1 event name to circuit_id matching Jolpica convention."""
    mapping = {
        "Australian Grand Prix": "albert_park",
        "Chinese Grand Prix": "shanghai",
        "Japanese Grand Prix": "suzuka",
        "Bahrain Grand Prix": "bahrain",
        "Saudi Arabian Grand Prix": "jeddah",
        "Miami Grand Prix": "miami",
        "Emilia Romagna Grand Prix": "imola",
        "Monaco Grand Prix": "monaco",
        "Spanish Grand Prix": "catalunya",
        "Canadian Grand Prix": "villeneuve",
        "Austrian Grand Prix": "red_bull_ring",
        "British Grand Prix": "silverstone",
        "Hungarian Grand Prix": "hungaroring",
        "Belgian Grand Prix": "spa",
        "Dutch Grand Prix": "zandvoort",
        "Italian Grand Prix": "monza",
        "Azerbaijan Grand Prix": "baku",
        "Singapore Grand Prix": "marina_bay",
        "United States Grand Prix": "americas",
        "Mexico City Grand Prix": "rodriguez",
        "São Paulo Grand Prix": "interlagos",
        "Las Vegas Grand Prix": "vegas",
        "Qatar Grand Prix": "losail",
        "Abu Dhabi Grand Prix": "yas_marina",
    }
    return mapping.get(event_name, event_name.lower().replace(" ", "_"))


def remove_placeholder_rows():
    """Remove any placeholder rows (status='Pending') from race_results."""
    rr = pd.read_parquet(DATA_DIR / "race_results.parquet")
    clean = rr[rr["status"] != "Pending"]
    if len(clean) < len(rr):
        clean.to_parquet(DATA_DIR / "race_results.parquet", index=False)
        logger.info(f"Cleaned {len(rr) - len(clean)} placeholder rows")


def detect_conditions(season: int, race_round: int) -> str:
    """Detect weather conditions from FastF1 session data. Returns 'dry', 'wet', or 'mixed'."""
    try:
        import fastf1
        session = fastf1.get_session(season, race_round, "R")
        session.load(telemetry=False, weather=True, messages=False)
        weather = session.weather_data
        if weather is not None and not weather.empty and "Rainfall" in weather.columns:
            rain_pct = weather["Rainfall"].mean()
            if rain_pct > 0.5:
                return "wet"
            elif rain_pct > 0.1:
                return "mixed"
    except Exception:
        pass
    return "dry"


def predict_race(
    season: int,
    race_round: int,
    n_simulations: int = 10000,
    conditions: str = "dry",
) -> pd.DataFrame:
    """
    Full prediction pipeline for an upcoming race.

    1. Loads trained model
    2. Loads feature matrix
    3. Extracts features for this race
    4. Predicts positions + probabilities
    5. Runs Monte Carlo simulation (condition-aware)

    Returns simulation results DataFrame.
    """
    from data.models.predictor import F1Predictor
    from data.models.simulator import RaceSimulator
    from data.features.elo import CIRCUIT_TYPES, DEFAULT_CIRCUIT_TYPE

    # Load model
    model = F1Predictor()
    model.load()

    # Load feature matrix
    fm = pd.read_parquet(DATA_DIR / "feature_matrix.parquet")

    # Get race data
    race_data = fm[
        (fm["season"] == season) & (fm["round"] == race_round)
    ]

    if race_data.empty:
        logger.error(
            "No features for %d Round %d. "
            "Run `python -m data.pipeline --step features` first.",
            season, race_round,
        )
        return pd.DataFrame()

    # Predict
    predictions = model.predict_race(race_data)

    # Determine circuit type for simulation
    circuit_id = race_data["circuit_id"].iloc[0]
    circuit_type = CIRCUIT_TYPES.get(circuit_id, DEFAULT_CIRCUIT_TYPE)

    # Build constructor map for correlated team DNFs
    constructor_map = None
    if "constructor_id" in race_data.columns:
        constructor_map = dict(zip(race_data["driver_id"], race_data["constructor_id"]))

    # Simulate with conditions and constructor correlation
    simulator = RaceSimulator(n_simulations=n_simulations)
    results = simulator.simulate_race(
        predictions,
        circuit_type,
        conditions=conditions,
        constructor_map=constructor_map,
    )

    return results


def format_prediction_table(results: pd.DataFrame) -> str:
    """Format prediction results as a readable table."""
    if results.empty:
        return "No predictions available."

    lines = []
    lines.append(f"{'Pos':>3}  {'Driver':<20} {'Pred Pos':>8} {'Win%':>6} "
                 f"{'Podium%':>7} {'Points%':>7} {'DNF%':>5} "
                 f"{'E[Pts]':>6} {'Med Pos':>7}")
    lines.append("-" * 85)

    for i, (_, row) in enumerate(results.iterrows(), 1):
        driver = row.get("driver_id", "???")
        lines.append(
            f"{i:>3}  {driver:<20} "
            f"{row.get('predicted_position', 0):>8.1f} "
            f"{row.get('sim_win_pct', 0):>5.1f}% "
            f"{row.get('sim_podium_pct', 0):>6.1f}% "
            f"{row.get('sim_points_pct', 0):>6.1f}% "
            f"{row.get('sim_dnf_pct', 0):>4.1f}% "
            f"{row.get('sim_expected_points', 0):>6.1f} "
            f"{row.get('sim_median_position', 0):>7.1f}"
        )

    return "\n".join(lines)


def run_weekend_prediction(
    season: int,
    race_round: int,
    refresh: bool = False,
    n_simulations: int = 10000,
    rebuild_features: bool = True,
):
    """
    End-to-end race weekend prediction:
    1. Fetch latest session data (FP1, Sprint Q, Sprint, Qualifying)
    2. Update FastF1 laps database
    3. Rebuild feature matrix (optional)
    4. Predict + simulate race
    5. Display results
    """
    import fastf1
    from data.ingest.fastf1_ingest import setup_fastf1_cache

    # Get event info
    setup_fastf1_cache()
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    event = schedule[schedule["RoundNumber"] == race_round]
    if event.empty:
        logger.error("Round %d not found in %d schedule", race_round, season)
        return

    event_name = event.iloc[0]["EventName"]
    print(f"\n{'='*60}")
    print(f"  {season} {event_name} — Race Prediction")
    print(f"{'='*60}\n")

    # Step 1: Fetch session data (may be empty for future races)
    print("Step 1: Fetching session data from FastF1...")
    weekend_laps = fetch_weekend_sessions(season, race_round, refresh=refresh)
    if weekend_laps.empty:
        print("  No session data available yet (race hasn't started).")
        print("  Predicting from historical features only.")
    else:
        sessions_found = weekend_laps["session_type"].unique()
        print(f"  Sessions loaded: {', '.join(sorted(sessions_found))}")
        print(f"  Total laps: {len(weekend_laps)}")

        # Step 2: Update FastF1 laps
        print("\nStep 2: Updating FastF1 laps database...")
        update_fastf1_laps(weekend_laps)

    # Step 3: Inject placeholder rows + rebuild feature matrix
    if rebuild_features:
        print("\nStep 3: Injecting upcoming race into dataset...")
        inject_upcoming_race(season, race_round)

        print("  Rebuilding feature matrix...")
        from data.pipeline import step_features
        step_features()

        # Clean up placeholder rows after feature extraction
        remove_placeholder_rows()

    # Step 4: Detect conditions and predict
    conditions = detect_conditions(season, race_round)
    print(f"\nStep 4: Conditions = {conditions.upper()}")
    print(f"  Running prediction + Monte Carlo ({n_simulations:,} simulations)...")
    results = predict_race(season, race_round, n_simulations=n_simulations, conditions=conditions)

    if results.empty:
        print("Prediction failed — check logs.")
        return

    # Step 5: Display
    print(f"\n{'='*60}")
    print(f"  RACE PREDICTION: {season} {event_name} [{conditions.upper()}]")
    print(f"{'='*60}\n")
    print(format_prediction_table(results))

    # Step 5b: Compare with betting odds (if available)
    try:
        from data.ingest.odds import OddsClient
        from data.models.value import ValueDetector

        client = OddsClient()
        odds = client.load_odds(season, race_round)
        if odds is not None and not odds.empty:
            detector = ValueDetector()
            value = detector.find_value(results, odds)
            if not value.empty:
                print(f"\n{'='*60}")
                print(f"  VALUE DETECTION (vs. bookmaker odds)")
                print(f"{'='*60}")
                for _, v in value.iterrows():
                    edge = v.get("edge", 0) * 100
                    rating = v.get("value_rating", "")
                    if edge > 0:
                        print(f"  {v['driver_id']:<20} edge={edge:+.1f}%  kelly={v.get('kelly_fraction', 0):.1%}  [{rating}]")
    except Exception:
        pass

    # Save results
    output_path = DATA_DIR / f"prediction_{season}_R{race_round:02d}.csv"
    save_cols = [
        "driver_id", "predicted_position", "predicted_rank",
        "prob_podium", "prob_winner", "prob_points", "prob_dnf",
        "sim_win_pct", "sim_podium_pct", "sim_points_pct", "sim_dnf_pct",
        "sim_expected_points", "sim_points_std",
        "sim_median_position", "sim_position_25", "sim_position_75",
    ]
    available = [c for c in save_cols if c in results.columns]
    results[available].to_csv(output_path, index=False)
    print(f"\nPrediction saved to {output_path}")

    # Sprint results summary
    sprint_results = get_sprint_results(season, race_round)
    if sprint_results is not None:
        print(f"\n{'='*60}")
        print(f"  SPRINT RESULTS (actual)")
        print(f"{'='*60}")
        for _, r in sprint_results.iterrows():
            print(f"  P{int(r['position']):>2}: {r['driver_code']}")

    # Qualifying summary
    quali_results = get_qualifying_results(season, race_round)
    if quali_results is not None:
        print(f"\n{'='*60}")
        print(f"  QUALIFYING RESULTS (actual)")
        print(f"{'='*60}")
        for _, r in quali_results.iterrows():
            pos = r.get("position")
            if pd.notna(pos):
                print(f"  P{int(pos):>2}: {r['driver_code']}  Q3={r.get('q3', '-')}")

    return results


def auto_detect_next_race() -> tuple[int, int]:
    """Detect the next upcoming race from the schedule.

    Skips rounds that already have race results in our data, so re-running
    after a race correctly targets the next one.
    """
    import fastf1
    from datetime import datetime
    from data.ingest.fastf1_ingest import setup_fastf1_cache

    setup_fastf1_cache()

    today = datetime.now()
    season = today.year

    # Check which rounds already have results (read only season/round columns)
    completed_rounds: set[int] = set()
    rr_path = DATA_DIR / "race_results.parquet"
    if rr_path.exists():
        rr = pd.read_parquet(rr_path, columns=["season", "round"])
        completed_rounds = set(
            rr[rr["season"] == season]["round"].unique().astype(int)
        )

    schedule = fastf1.get_event_schedule(season, include_testing=False)
    for _, event in schedule.iterrows():
        rnd = event.get("RoundNumber", 0)
        if rnd == 0:
            continue
        rnd = int(rnd)
        if rnd in completed_rounds:
            continue
        event_date = pd.to_datetime(event["EventDate"])
        if event_date.date() >= today.date():
            return season, rnd

    # If all races passed, return last round
    last = schedule[schedule["RoundNumber"] > 0].iloc[-1]
    return season, int(last["RoundNumber"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Race Weekend Predictor")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch session data")
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--no-rebuild", action="store_true", help="Skip feature matrix rebuild")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Suppress noisy FastF1 logs before any imports
    for noisy in ["fastf1", "req", "core", "fastf1.core", "fastf1.req",
                   "fastf1._api", "fastf1.logger"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.season and args.round:
        season, race_round = args.season, args.round
    else:
        season, race_round = auto_detect_next_race()
        print(f"Auto-detected next race: {season} Round {race_round}")

    run_weekend_prediction(
        season=season,
        race_round=race_round,
        refresh=args.refresh,
        n_simulations=args.simulations,
        rebuild_features=not args.no_rebuild,
    )
