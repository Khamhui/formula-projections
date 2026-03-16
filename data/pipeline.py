"""
Main F1 prediction pipeline — end-to-end from data ingestion to trained model.

Usage:
    python -m data.pipeline --step ingest      # Download all historical data
    python -m data.pipeline --step features    # Build feature matrix
    python -m data.pipeline --step train       # Train prediction models
    python -m data.pipeline --step predict     # Predict next race
    python -m data.pipeline --step calibrate   # Verify model calibration
    python -m data.pipeline --step odds        # Fetch betting odds
    python -m data.pipeline --step all         # Run everything (except calibrate/odds)
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "cache" / "processed"


def step_ingest(start_year: int = 1950, end_year: int = 2025, merge: bool = False):
    """Step 1: Download historical data from Jolpica + FastF1.

    Args:
        merge: If True, merge new data into existing parquet files instead of
               overwriting. Used by auto_update to re-ingest a single season
               without losing historical data.
    """
    from data.ingest.jolpica import JolpicaClient

    logger.info("=" * 60)
    logger.info("STEP 1: Data Ingestion")
    logger.info("=" * 60)

    # Jolpica: historical backbone
    logger.info("Ingesting from Jolpica (%d–%d)...", start_year, end_year)
    client = JolpicaClient(cache=True)
    jolpica_data = client.ingest_all_seasons(
        start_year=start_year, end_year=end_year,
        save_parquet=not merge,  # Don't save if merging — we'll save after merge
    )

    if merge:
        jolpica_data = _merge_with_existing(jolpica_data, start_year, end_year)

    for name, df in jolpica_data.items():
        logger.info(f"  {name}: {len(df):,} rows")

    # FastF1: 2018+ (granular lap data — optional, takes longer)
    try:
        from data.ingest.fastf1_ingest import ingest_all as fastf1_ingest

        logger.info("\nIngesting from FastF1 (2018+)...")
        f1_start = max(start_year, 2018)
        fastf1_data = fastf1_ingest(start_year=f1_start, end_year=end_year)

        # Merge FastF1 laps with existing data when in merge mode
        if merge and "laps" in fastf1_data:
            laps_path = DATA_DIR / "fastf1_laps.parquet"
            if laps_path.exists():
                existing_laps = pd.read_parquet(laps_path)
                keep = existing_laps[~existing_laps["year"].between(f1_start, end_year)]
                combined = pd.concat([keep, fastf1_data["laps"]], ignore_index=True)
                combined = combined.sort_values(["year", "gp"])
                combined.to_parquet(laps_path, index=False)
                fastf1_data["laps"] = combined
                logger.info(
                    "  Merged fastf1_laps: kept %d historic + %d new = %d total",
                    len(keep), len(fastf1_data["laps"]) - len(keep), len(combined),
                )

        for name, df in fastf1_data.items():
            logger.info(f"  {name}: {len(df):,} rows")
    except ImportError:
        logger.warning("FastF1 not installed. Skipping granular lap data.")
    except Exception as e:
        logger.warning(f"FastF1 ingestion failed: {e}. Continuing with Jolpica data.")

    logger.info("\nIngestion complete.")


def _merge_with_existing(
    new_data: dict[str, pd.DataFrame],
    start_year: int,
    end_year: int,
) -> dict[str, pd.DataFrame]:
    """Merge freshly ingested season data with existing parquet files.

    Replaces rows for the ingested year range, keeps all other years intact.
    """
    merged = {}
    for name, new_df in new_data.items():
        existing_path = DATA_DIR / f"{name}.parquet"
        if existing_path.exists() and "season" in new_df.columns:
            existing = pd.read_parquet(existing_path)
            # Remove old data for the re-ingested seasons
            keep = existing[
                ~existing["season"].between(start_year, end_year)
            ]
            combined = pd.concat([keep, new_df], ignore_index=True)
            combined = combined.sort_values(
                ["season", "round"] if "round" in combined.columns else ["season"],
            )
            combined.to_parquet(existing_path, index=False)
            merged[name] = combined
            logger.info(
                "  Merged %s: kept %d historic + %d new = %d total",
                name, len(keep), len(new_df), len(combined),
            )
        else:
            merged[name] = new_df
    return merged


def step_features():
    """Step 2: Build feature matrix from ingested data."""
    from data.features.engineer import build_feature_matrix

    logger.info("=" * 60)
    logger.info("STEP 2: Feature Engineering")
    logger.info("=" * 60)

    # Load ingested data
    race_results = pd.read_parquet(DATA_DIR / "race_results.parquet")

    qualifying = None
    quali_path = DATA_DIR / "qualifying.parquet"
    if quali_path.exists():
        qualifying = pd.read_parquet(quali_path)

    fastf1_laps = None
    fastf1_path = DATA_DIR / "fastf1_laps.parquet"
    if fastf1_path.exists():
        fastf1_laps = pd.read_parquet(fastf1_path)

    sprints = None
    sprints_path = DATA_DIR / "sprints.parquet"
    if sprints_path.exists():
        sprints = pd.read_parquet(sprints_path)

    # Build features
    feature_matrix = build_feature_matrix(
        race_results=race_results,
        qualifying=qualifying,
        fastf1_laps=fastf1_laps,
        sprints=sprints,
    )

    # Save
    feature_matrix.to_parquet(DATA_DIR / "feature_matrix.parquet", index=False)
    logger.info(f"Feature matrix saved: {len(feature_matrix):,} rows × {len(feature_matrix.columns)} columns")

    return feature_matrix


def step_train(test_seasons: list[int] = None):
    """Step 3: Train XGBoost models."""
    from data.models.predictor import train_and_evaluate

    logger.info("=" * 60)
    logger.info("STEP 3: Model Training")
    logger.info("=" * 60)

    feature_matrix = pd.read_parquet(DATA_DIR / "feature_matrix.parquet")

    model, metrics = train_and_evaluate(feature_matrix, test_seasons=test_seasons)

    # Save model
    model.save()

    logger.info("\nTraining complete.")
    logger.info(f"Position MAE: {metrics.get('position_mae', 'N/A')}")
    logger.info(f"Podium accuracy: {metrics.get('podium_accuracy', 'N/A')}")
    logger.info(f"Winner accuracy: {metrics.get('winner_accuracy', 'N/A')}")

    return model, metrics


def step_predict():
    """Step 4: Predict upcoming race results."""
    from data.models.predictor import F1Predictor

    logger.info("=" * 60)
    logger.info("STEP 4: Prediction")
    logger.info("=" * 60)

    # Load model and data
    model = F1Predictor()
    model.load()

    feature_matrix = pd.read_parquet(DATA_DIR / "feature_matrix.parquet")

    # Use latest race data to predict next race
    latest = feature_matrix[feature_matrix["season"] == feature_matrix["season"].max()]
    latest_round = latest["round"].max()

    logger.info(f"Latest data: Season {latest['season'].max()}, Round {latest_round}")
    logger.info("To predict a specific upcoming race, use the notebook interface.")


def step_calibrate(test_seasons: list[int] = None):
    """Step 5: Evaluate model calibration on historical predictions."""
    from data.models.calibration import evaluate_model_calibration, print_calibration_report

    logger.info("=" * 60)
    logger.info("STEP 5: Calibration Verification")
    logger.info("=" * 60)

    feature_matrix = pd.read_parquet(DATA_DIR / "feature_matrix.parquet")

    if test_seasons is None:
        max_season = int(feature_matrix["season"].max())
        test_seasons = [max_season - 1, max_season]
        logger.info(f"No test seasons specified, using {test_seasons}")

    report = evaluate_model_calibration(feature_matrix, test_seasons=test_seasons)
    print_calibration_report(report)

    logger.info("\nCalibration verification complete.")
    return report


def step_odds(season: int = None, race_round: int = None):
    """Step 6: Fetch and store betting odds for value detection."""
    from data.ingest.odds import OddsClient

    logger.info("=" * 60)
    logger.info("STEP 6: Odds Ingestion")
    logger.info("=" * 60)

    client = OddsClient()

    if season and race_round:
        odds = client.fetch_race_winner_odds(season, race_round)
        if odds is not None and not odds.empty:
            client.save_odds(odds, season, race_round)
            logger.info(f"Saved odds for {season} Round {race_round}: {len(odds)} drivers")
        else:
            logger.warning("No odds available from API. Use --odds-csv for CSV import.")
    else:
        odds = client.fetch_current_odds()
        if odds is not None and not odds.empty:
            logger.info(f"Current odds fetched: {len(odds)} drivers")
        else:
            logger.warning("No current odds available.")

    return odds


def step_deep_train(n_epochs: int = 50):
    """Step 7: Train temporal deep learning model for embedding extraction."""
    logger.info("=" * 60)
    logger.info("STEP 7: Temporal Deep Learning Training")
    logger.info("=" * 60)

    try:
        from data.models.deep.temporal_model import train_temporal_model
    except ImportError:
        logger.warning("PyTorch not installed. Skipping deep learning step.")
        logger.warning("Install with: pip install torch>=2.0.0")
        return None

    feature_matrix = pd.read_parquet(DATA_DIR / "feature_matrix.parquet")
    trainer = train_temporal_model(feature_matrix, n_epochs=n_epochs)

    if trainer:
        logger.info("Temporal model training complete.")
    else:
        logger.warning("Temporal model training skipped.")

    return trainer


def step_championship(season: int = None, n_simulations: int = 10000):
    """Step 8: Simulate remaining championship season."""
    from data.models.championship import (
        ChampionshipSimulator, remaining_calendar, load_current_standings,
    )
    from data.models.predictor import F1Predictor

    logger.info("=" * 60)
    logger.info("STEP 8: Championship Monte Carlo")
    logger.info("=" * 60)

    fm = pd.read_parquet(DATA_DIR / "feature_matrix.parquet")

    if season is None:
        season = int(fm["season"].max())

    calendar = remaining_calendar(season)
    if not calendar:
        logger.warning("No remaining races found for %d", season)
        return None

    logger.info("Remaining races: %d", len(calendar))

    driver_standings, constructor_standings = load_current_standings(season)

    model = F1Predictor()
    model.load()
    latest_data = fm[fm["season"] == season]
    latest_round = int(latest_data["round"].max()) if not latest_data.empty else 0

    race_predictions = []
    circuit_types = []
    race_names = []
    sprint_list = []

    for race in calendar:
        race_data = fm[(fm["season"] == season) & (fm["round"] == latest_round)]
        if race_data.empty:
            continue
        pred = model.predict_race(race_data)
        race_predictions.append(pred)
        circuit_types.append("mixed")
        race_names.append(race["name"])
        sprint_list.append(race.get("is_sprint", False))

    if not race_predictions:
        logger.warning("No predictions generated")
        return None

    constructor_map = {}
    if "constructor_id" in fm.columns:
        latest = fm[(fm["season"] == season) & (fm["round"] == latest_round)]
        for _, row in latest.iterrows():
            constructor_map[row["driver_id"]] = row["constructor_id"]

    sim = ChampionshipSimulator(n_simulations=n_simulations)
    result = sim.simulate_season(
        race_predictions=race_predictions,
        circuit_types=circuit_types,
        race_names=race_names,
        current_standings=driver_standings,
        constructor_standings=constructor_standings,
        constructor_map=constructor_map,
        sprint_races=sprint_list,
    )

    wdc = result["wdc"]
    if not wdc.empty:
        wdc.to_csv(DATA_DIR / f"championship_wdc_{season}.csv", index=False)
        logger.info("Championship WDC saved: %d drivers", len(wdc))

    wcc = result.get("wcc")
    if wcc is not None and not wcc.empty:
        wcc.to_csv(DATA_DIR / f"championship_wcc_{season}.csv", index=False)
        logger.info("Championship WCC saved: %d constructors", len(wcc))

    return result


def step_alpha(test_seasons: list[int] = None):
    """Step 9: Run market alpha analysis on backtest results."""
    from data.models.alpha import backtest_alpha

    logger.info("=" * 60)
    logger.info("STEP 9: Market Alpha Analysis")
    logger.info("=" * 60)

    pred_path = DATA_DIR / "backtest_driver_predictions.csv"
    if not pred_path.exists():
        logger.warning("No backtest predictions found. Run backtest first.")
        return None

    predictions = pd.read_csv(pred_path)
    fm = pd.read_parquet(DATA_DIR / "feature_matrix.parquet")

    result = backtest_alpha(predictions, fm)

    summary = result.get("summary", {})
    if summary:
        logger.info("Alpha report: %d races with odds", summary.get("races_with_odds", 0))
        logger.info("  Model Brier: %.4f", summary.get("mean_brier_model", 0))
        logger.info("  Market Brier: %.4f", summary.get("mean_brier_market", 0))
        logger.info("  Alpha: %.4f", summary.get("mean_alpha_brier", 0))
        logger.info("  Kelly ROI: %.2f%%", summary.get("roi", 0) * 100)

    per_race = result.get("per_race")
    if per_race is not None and not per_race.empty:
        per_race.to_csv(DATA_DIR / "alpha_per_race.csv", index=False)

    return result


def main():
    parser = argparse.ArgumentParser(description="F1 Prediction Pipeline")
    parser.add_argument(
        "--step",
        choices=[
            "ingest", "features", "train", "predict", "calibrate", "odds",
            "deep-train", "championship", "alpha", "all",
        ],
        default="all",
        help="Pipeline step to run",
    )
    parser.add_argument("--start-year", type=int, default=2003)
    parser.add_argument("--end-year", type=int, default=datetime.now().year)
    parser.add_argument("--test-seasons", type=int, nargs="+", default=None)
    parser.add_argument("--odds-season", type=int, default=None)
    parser.add_argument("--odds-round", type=int, default=None)
    parser.add_argument("--championship-season", type=int, default=None)
    parser.add_argument("--deep-epochs", type=int, default=50)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.step in ("ingest", "all"):
        step_ingest(args.start_year, args.end_year)

    if args.step in ("features", "all"):
        step_features()

    if args.step in ("train", "all"):
        step_train(args.test_seasons)

    if args.step in ("predict", "all"):
        step_predict()

    if args.step == "calibrate":
        step_calibrate(args.test_seasons)

    if args.step == "odds":
        step_odds(args.odds_season, args.odds_round)

    if args.step == "deep-train":
        step_deep_train(n_epochs=args.deep_epochs)

    if args.step == "championship":
        step_championship(season=args.championship_season)

    if args.step == "alpha":
        step_alpha(args.test_seasons)


if __name__ == "__main__":
    main()
