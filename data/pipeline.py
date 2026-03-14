"""
Main F1 prediction pipeline — end-to-end from data ingestion to trained model.

Usage:
    python -m data.pipeline --step ingest     # Download all historical data
    python -m data.pipeline --step features   # Build feature matrix
    python -m data.pipeline --step train      # Train prediction models
    python -m data.pipeline --step all        # Run everything
    python -m data.pipeline --step predict    # Predict next race
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "cache" / "processed"


def step_ingest(start_year: int = 1950, end_year: int = 2025):
    """Step 1: Download all historical data from Jolpica + FastF1."""
    from data.ingest.jolpica import JolpicaClient

    logger.info("=" * 60)
    logger.info("STEP 1: Data Ingestion")
    logger.info("=" * 60)

    # Jolpica: 1950–present (historical backbone)
    logger.info("Ingesting from Jolpica (1950–present)...")
    client = JolpicaClient(cache=True)
    jolpica_data = client.ingest_all_seasons(start_year=start_year, end_year=end_year)

    for name, df in jolpica_data.items():
        logger.info(f"  {name}: {len(df):,} rows")

    # FastF1: 2018+ (granular lap data — optional, takes longer)
    try:
        from data.ingest.fastf1_ingest import ingest_all as fastf1_ingest

        logger.info("\nIngesting from FastF1 (2018+)...")
        fastf1_data = fastf1_ingest(start_year=max(start_year, 2018), end_year=end_year)
        for name, df in fastf1_data.items():
            logger.info(f"  {name}: {len(df):,} rows")
    except ImportError:
        logger.warning("FastF1 not installed. Skipping granular lap data.")
    except Exception as e:
        logger.warning(f"FastF1 ingestion failed: {e}. Continuing with Jolpica data.")

    logger.info("\nIngestion complete.")


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


def main():
    parser = argparse.ArgumentParser(description="F1 Prediction Pipeline")
    parser.add_argument(
        "--step",
        choices=["ingest", "features", "train", "predict", "all"],
        default="all",
        help="Pipeline step to run",
    )
    parser.add_argument("--start-year", type=int, default=2003)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--test-seasons", type=int, nargs="+", default=None)
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


if __name__ == "__main__":
    main()
