"""
F1 Race Prediction Model — CatBoost / XGBoost / LightGBM + Stacking Ensemble.

Prediction targets:
1. Race finishing position (stacking ensemble regression)
2. Podium finish (calibrated binary classification)
3. Points finish (calibrated binary classification)
4. Race winner (calibrated binary classification)
5. DNF probability (calibrated binary classification)

Based on the tennis prediction approach but adapted for F1's unique
characteristics (constructor performance, grid position, tire strategy).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "cache" / "models"


def create_model(task: str = "regressor", **kwargs):
    """
    Create the best available gradient boosting model.
    Tries CatBoost -> XGBoost -> LightGBM -> sklearn HistGradientBoosting.
    task: "regressor" or "classifier"
    """
    is_reg = task == "regressor"

    # Try CatBoost first (ships own OpenMP, no external deps)
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier
        cb_params = {
            "iterations": kwargs.get("n_estimators", 500),
            "depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "subsample": kwargs.get("subsample", 0.8),
            "l2_leaf_reg": kwargs.get("reg_lambda", 1.0),
            "random_seed": kwargs.get("random_state", 42),
            "verbose": 0,
            "thread_count": -1,
        }
        if not is_reg and "scale_pos_weight" in kwargs:
            cb_params["scale_pos_weight"] = float(kwargs["scale_pos_weight"])
        cls = CatBoostRegressor if is_reg else CatBoostClassifier
        logger.info(f"Using CatBoost{cls.__name__}")
        return cls(**cb_params)
    except ImportError:
        pass

    # Try XGBoost
    try:
        import xgboost as xgb
        cls = xgb.XGBRegressor if is_reg else xgb.XGBClassifier
        logger.info(f"Using {cls.__name__}")
        return cls(**kwargs)
    except ImportError:
        pass

    # Try LightGBM
    try:
        import lightgbm as lgb
        lgb_params = {
            "n_estimators": kwargs.get("n_estimators", 500),
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "reg_alpha": kwargs.get("reg_alpha", 0.1),
            "reg_lambda": kwargs.get("reg_lambda", 1.0),
            "min_child_samples": kwargs.get("min_child_weight", 5),
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": kwargs.get("n_jobs", -1),
            "verbose": -1,
        }
        if "scale_pos_weight" in kwargs:
            lgb_params["scale_pos_weight"] = kwargs["scale_pos_weight"]
        cls = lgb.LGBMRegressor if is_reg else lgb.LGBMClassifier
        logger.info(f"Using {cls.__name__}")
        return cls(**lgb_params)
    except ImportError:
        pass

    # Fallback: sklearn HistGradientBoosting (no OpenMP dependency)
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    logger.info(f"Using sklearn HistGradientBoosting{task.title()} (install libomp for XGBoost/LightGBM)")
    cls = HistGradientBoostingRegressor if is_reg else HistGradientBoostingClassifier
    return cls(
        max_iter=kwargs.get("n_estimators", 500),
        max_depth=kwargs.get("max_depth", 6),
        learning_rate=kwargs.get("learning_rate", 0.05),
        random_state=kwargs.get("random_state", 42),
    )


class F1Predictor:
    """
    Multi-target F1 prediction model.

    Uses LightGBM/XGBoost with time-series cross-validation to avoid data leakage.
    """

    def __init__(self):
        self.position_model = None
        self.podium_model = None
        self.winner_model = None
        self.points_model = None
        self.dnf_model = None
        self.feature_names: list[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        self._feature_medians: Optional[pd.Series] = None

    def train(self, X: pd.DataFrame, y_position: pd.Series, y_dnf: Optional[pd.Series] = None):
        """Train all prediction models with time-series cross-validation."""
        if len(X) < 6:
            raise ValueError(
                f"Training requires at least 6 samples, got {len(X)}. "
                "Check that the feature matrix has enough historical data."
            )

        self.feature_names = list(X.columns)

        # Replace inf with NaN, then fill NaN with column medians
        import numpy as np
        X = X.replace([np.inf, -np.inf], np.nan)
        self._feature_medians = X.median()
        X = X.fillna(self._feature_medians).fillna(0)
        self._feature_medians = self._feature_medians.fillna(0)

        y_podium = (y_position <= 3).astype(int)
        y_winner = (y_position == 1).astype(int)
        y_points = (y_position <= 10).astype(int)

        n_splits = max(2, min(5, len(X) - 1))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        # StackingRegressor requires partitioning CV (every sample in exactly one fold).
        # TimeSeriesSplit doesn't partition — it expands the training set each fold.
        # KFold(shuffle=False) preserves temporal order while partitioning.
        from sklearn.model_selection import KFold
        stacking_cv = KFold(n_splits=n_splits, shuffle=False)

        def _calibrate(base, y_binary):
            """Wrap classifier in calibrator. Falls back gracefully if calibration fails."""
            min_class = min(y_binary.sum(), len(y_binary) - y_binary.sum())
            if min_class < 5:
                logger.warning("Too few samples for calibration (min class=%d), using uncalibrated", min_class)
                return base
            try:
                from data.models.venn_abers import VennAbersCalibrator
                return VennAbersCalibrator(base, cal_fraction=0.3)
            except Exception:
                pass
            try:
                splits = max(2, min(3, int(min_class // 2)))
                return CalibratedClassifierCV(base, cv=splits, method="isotonic")
            except Exception as e:
                logger.warning("All calibration failed (%s), using uncalibrated model", e)
                return base

        # Load Optuna-tuned params if available
        tuned = None
        try:
            from data.models.tuner import load_tuned_params
            tuned = load_tuned_params()
            if tuned:
                logger.info("Using Optuna-tuned hyperparameters")
        except ImportError:
            pass
        except Exception as e:
            logger.warning("Failed to load tuned params: %s", e)

        def _get_params(model_name: str, defaults: dict) -> dict:
            if tuned and model_name in tuned:
                p = tuned[model_name]
                return dict(
                    n_estimators=p.get("n_estimators", defaults.get("n_estimators", 500)),
                    max_depth=p.get("max_depth", defaults.get("max_depth", 6)),
                    learning_rate=p.get("learning_rate", defaults.get("learning_rate", 0.05)),
                    subsample=p.get("subsample", defaults.get("subsample", 0.8)),
                    min_child_weight=p.get("min_child_weight", defaults.get("min_child_weight", 5)),
                    reg_alpha=p.get("reg_alpha", defaults.get("reg_alpha", 0.1)),
                    reg_lambda=p.get("reg_lambda", defaults.get("reg_lambda", 1.0)),
                    random_state=42, n_jobs=-1,
                )
            return {**defaults, "random_state": 42, "n_jobs": -1}

        pos_defaults = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                           subsample=0.8, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0)
        pos_params = _get_params("position", pos_defaults)

        # 1. Position Regression — Stacking Ensemble
        logger.info("Training stacking position regression model...")
        base_gb = create_model("regressor", **pos_params)
        base_extra = ExtraTreesRegressor(
            n_estimators=300, max_depth=8, random_state=42, n_jobs=-1,
        )

        self.position_model = StackingRegressor(
            estimators=[("gb", base_gb), ("extra", base_extra)],
            final_estimator=Ridge(alpha=10.0),
            cv=stacking_cv,
            passthrough=False,
            n_jobs=-1,
        )

        # Quick single-model CV baseline
        _quick = create_model("regressor", **pos_params)
        cv_scores = cross_val_score(
            _quick, X, y_position, cv=tscv, scoring="neg_mean_absolute_error",
        )
        logger.info(f"Position MAE (CV, single model baseline): {-cv_scores.mean():.2f} +/- {cv_scores.std():.2f}")

        self.position_model.fit(X, y_position)
        logger.info("Stacking ensemble fitted (GB + ExtraTrees → Ridge)")

        # 2. Podium Classification (calibrated)
        logger.info("Training calibrated podium classifier...")
        pod_defaults = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                           min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0)
        pod_params = _get_params("podium", pod_defaults)
        pod_params["scale_pos_weight"] = len(y_podium) / max(y_podium.sum(), 1) - 1
        _podium_base = create_model("classifier", **pod_params)
        cv_scores = cross_val_score(_podium_base, X, y_podium, cv=tscv, scoring="accuracy")
        logger.info(f"Podium accuracy (CV): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        self.podium_model = _calibrate(_podium_base, y_podium)
        self.podium_model.fit(X, y_podium)

        # 3. Winner Classification (calibrated)
        logger.info("Training calibrated winner classifier...")
        win_defaults = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
                           min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0)
        win_params = _get_params("winner", win_defaults)
        win_params["scale_pos_weight"] = len(y_winner) / max(y_winner.sum(), 1) - 1
        _winner_base = create_model("classifier", **win_params)
        self.winner_model = _calibrate(_winner_base, y_winner)
        self.winner_model.fit(X, y_winner)

        # 4. Points Classification (calibrated)
        logger.info("Training calibrated points classifier...")
        pts_defaults = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                           min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0)
        pts_params = _get_params("points", pts_defaults)
        _points_base = create_model("classifier", **pts_params)
        self.points_model = _calibrate(_points_base, y_points)
        self.points_model.fit(X, y_points)

        # 5. DNF Classification (calibrated)
        logger.info("Training calibrated DNF classifier...")
        if y_dnf is None:
            y_dnf = (y_position.isna() | (y_position > 20)).astype(int)
        else:
            y_dnf = y_dnf.reindex(X.index).fillna(0).astype(int)
        _dnf_base = create_model(
            "classifier", n_estimators=300, max_depth=4, learning_rate=0.03,
            scale_pos_weight=len(y_dnf) / max(y_dnf.sum(), 1) - 1,
            subsample=0.8, random_state=42, n_jobs=-1,
        )
        cv_dnf = cross_val_score(_dnf_base, X, y_dnf, cv=tscv, scoring="accuracy")
        logger.info(f"DNF accuracy (CV): {cv_dnf.mean():.3f} +/- {cv_dnf.std():.3f}")
        self.dnf_model = _calibrate(_dnf_base, y_dnf)
        self.dnf_model.fit(X, y_dnf)

        # Feature importance — stacking model needs permutation importance
        if hasattr(self.position_model, "feature_importances_"):
            importance = self.position_model.feature_importances_
        else:
            from sklearn.inspection import permutation_importance
            perm = permutation_importance(self.position_model, X, y_position, n_repeats=5, random_state=42)
            importance = perm.importances_mean

        self.feature_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        logger.info("Top 10 features:")
        for _, row in self.feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    def select_features(self, X: pd.DataFrame, threshold: float = 0.001) -> list[str]:
        """Return features with importance above threshold."""
        if self.feature_importance is None:
            return list(X.columns)
        selected = self.feature_importance[
            self.feature_importance["importance"] >= threshold
        ]["feature"].tolist()
        dropped = len(self.feature_importance) - len(selected)
        logger.info(
            f"Feature selection: {len(selected)}/{len(self.feature_importance)} "
            f"features above {threshold} ({dropped} dropped)"
        )
        return selected

    def _fill_nan(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN/inf with training medians for models that don't handle missing values."""
        import numpy as np
        X = X.replace([np.inf, -np.inf], np.nan)
        if not X.isna().any().any():
            return X
        if self._feature_medians is not None:
            return X.fillna(self._feature_medians).fillna(0)
        return X.fillna(0)

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align columns to training features, adding missing ones as NaN."""
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            X = X.copy()
            for col in missing:
                X[col] = np.nan
        return self._fill_nan(X[self.feature_names])

    def predict_race(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict a full race grid with positions and probabilities."""
        if self.position_model is None:
            raise ValueError("Models not trained. Call train() first.")

        results = X.copy()
        features = self._align_features(X)

        results["predicted_position"] = self.position_model.predict(features)

        # For Venn-ABERS models, extract both probabilities and intervals from
        # a single _raw_intervals call to avoid redundant base model inference
        for name, model in [("winner", self.winner_model), ("podium", self.podium_model),
                            ("points", self.points_model), ("dnf", self.dnf_model)]:
            if model is None:
                continue
            if hasattr(model, "_raw_intervals"):
                try:
                    p0, p1 = model._raw_intervals(features)
                    denom = np.where((1.0 - p0) + p1 == 0, 1e-10, (1.0 - p0) + p1)
                    results[f"prob_{name}"] = np.clip(p1 / denom, 0.0, 1.0)
                    results[f"prob_{name}_lo"] = np.clip(np.minimum(p0, p1), 0.0, 1.0)
                    results[f"prob_{name}_hi"] = np.clip(np.maximum(p0, p1), 0.0, 1.0)
                except Exception as e:
                    logger.warning("Venn-ABERS inference failed for %s: %s", name, e)
                    results[f"prob_{name}"] = model.predict_proba(features)[:, 1]
            else:
                results[f"prob_{name}"] = model.predict_proba(features)[:, 1]

        results["predicted_rank"] = results["predicted_position"].rank().astype(int)

        return results.sort_values("predicted_position")

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_position: pd.Series,
        race_groups: Optional[pd.Series] = None,
        y_dnf: Optional[pd.Series] = None,
    ) -> dict:
        """Evaluate model on test data."""
        features = self._align_features(X_test)
        pred_pos = self.position_model.predict(features)
        pred_podium = self.podium_model.predict(features)
        pred_winner = self.winner_model.predict(features)

        y_podium = (y_position <= 3).astype(int)
        y_winner = (y_position == 1).astype(int)

        metrics = {
            "position_mae": mean_absolute_error(y_position, pred_pos),
            "podium_accuracy": accuracy_score(y_podium, pred_podium),
            "winner_accuracy": accuracy_score(y_winner, pred_winner),
        }

        if self.dnf_model is not None and y_dnf is not None:
            y_dnf_aligned = y_dnf.reindex(X_test.index).fillna(0).astype(int)
            pred_dnf = self.dnf_model.predict(features)
            metrics["dnf_accuracy"] = accuracy_score(y_dnf_aligned, pred_dnf)
            # DNF recall — how many actual DNFs did we predict
            actual_dnfs = y_dnf_aligned.sum()
            if actual_dnfs > 0:
                metrics["dnf_recall"] = (pred_dnf[y_dnf_aligned == 1] == 1).sum() / actual_dnfs

        logger.info("Evaluation results:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        return metrics

    def save(self, path: Optional[Path] = None):
        """Save all models to disk."""
        path = path or MODEL_DIR
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.position_model, path / "position_model.joblib")
        joblib.dump(self.podium_model, path / "podium_model.joblib")
        joblib.dump(self.winner_model, path / "winner_model.joblib")
        joblib.dump(self.points_model, path / "points_model.joblib")
        if self.dnf_model is not None:
            joblib.dump(self.dnf_model, path / "dnf_model.joblib")
        joblib.dump(self.feature_names, path / "feature_names.joblib")
        if self._feature_medians is not None:
            joblib.dump(self._feature_medians, path / "feature_medians.joblib")

        if self.feature_importance is not None:
            self.feature_importance.to_csv(path / "feature_importance.csv", index=False)

        logger.info(f"Models saved to {path}")

    def load(self, path: Optional[Path] = None):
        """Load models from disk."""
        path = path or MODEL_DIR

        self.position_model = joblib.load(path / "position_model.joblib")
        self.podium_model = joblib.load(path / "podium_model.joblib")
        self.winner_model = joblib.load(path / "winner_model.joblib")
        self.points_model = joblib.load(path / "points_model.joblib")
        self.feature_names = joblib.load(path / "feature_names.joblib")

        dnf_path = path / "dnf_model.joblib"
        if dnf_path.exists():
            self.dnf_model = joblib.load(dnf_path)

        medians_path = path / "feature_medians.joblib"
        if medians_path.exists():
            self._feature_medians = joblib.load(medians_path)

        importance_path = path / "feature_importance.csv"
        if importance_path.exists():
            self.feature_importance = pd.read_csv(importance_path)

        logger.info(f"Models loaded from {path}")


def _fit_pl_model(feature_matrix: pd.DataFrame):
    """
    Fit a Plackett-Luce model on the training split.

    Returns the fitted model, or None if fitting fails or data is insufficient.
    """
    try:
        from data.models.plackett_luce import PlackettLuceModel

        required = {"season", "round", "driver_id", "constructor_id", "position"}
        if not required.issubset(feature_matrix.columns):
            logger.warning("Missing columns for PL features, skipping")
            return None

        race_data = feature_matrix[["season", "round", "driver_id", "constructor_id", "position"]].dropna()
        if len(race_data) < 100:
            logger.warning("Too few races for PL (%d), skipping", len(race_data))
            return None

        pl = PlackettLuceModel()
        pl.fit(race_data)
        return pl

    except Exception as e:
        logger.warning("PL model fitting failed: %s", e)
        return None


def _inject_pl_features(feature_matrix: pd.DataFrame, X: pd.DataFrame, pl_model=None) -> pd.DataFrame:
    """
    Inject Plackett-Luce features into the feature matrix for ensemble stacking.

    Uses a pre-fitted PL model to add:
    pl_driver_strength, pl_constructor_strength, pl_combined_strength
    """
    if pl_model is None:
        return X

    try:
        # Vectorized PL feature lookup via pandas map
        shared_idx = X.index.intersection(feature_matrix.index)
        drivers = feature_matrix.loc[shared_idx, "driver_id"].astype(str)
        constructors = feature_matrix.loc[shared_idx, "constructor_id"].astype(str)

        X = X.copy()
        X["pl_driver_strength"] = drivers.map(pl_model.driver_strengths).fillna(0.0).reindex(X.index, fill_value=0.0)
        X["pl_constructor_strength"] = constructors.map(pl_model.constructor_strengths).fillna(0.0).reindex(X.index, fill_value=0.0)
        X["pl_combined_strength"] = X["pl_driver_strength"] + X["pl_constructor_strength"]

        logger.info("Injected 3 Plackett-Luce features into feature matrix")
        return X

    except Exception as e:
        logger.warning("PL feature injection failed: %s", e)
        return X


def train_and_evaluate(
    feature_matrix: pd.DataFrame,
    test_seasons: list[int] = None,
) -> tuple[F1Predictor, dict]:
    """Full training pipeline with train/test split by season."""
    from data.features.engineer import prepare_training_data

    if test_seasons is None:
        max_season = int(feature_matrix["season"].max())
        all_seasons = sorted(feature_matrix["season"].unique())
        if len(all_seasons) >= 3:
            test_seasons = [max_season - 1, max_season]
        elif len(all_seasons) == 2:
            test_seasons = [max_season]
        else:
            test_seasons = []

    train_mask = ~feature_matrix["season"].isin(test_seasons)
    test_mask = feature_matrix["season"].isin(test_seasons)

    # Extract DNF labels BEFORE prepare_training_data() drops rows
    train_dnf_all = feature_matrix[train_mask]["dnf"].dropna().astype(int)
    test_dnf_all = feature_matrix[test_mask]["dnf"].dropna().astype(int)

    X_train, y_train = prepare_training_data(
        feature_matrix[train_mask], target="position"
    )
    X_test, y_test = prepare_training_data(
        feature_matrix[test_mask], target="position"
    )

    # Inject Plackett-Luce features (trained on training data only, fit once)
    pl_model = _fit_pl_model(feature_matrix[train_mask])
    X_train = _inject_pl_features(feature_matrix[train_mask], X_train, pl_model)
    X_test = _inject_pl_features(feature_matrix[train_mask], X_test, pl_model)

    # Align columns
    for col in set(X_train.columns) - set(X_test.columns):
        X_test[col] = 0
    X_test = X_test[X_train.columns]

    # Align DNF labels with post-filtered indices
    train_dnf = train_dnf_all.reindex(X_train.index).fillna(0).astype(int)
    test_dnf = test_dnf_all.reindex(X_test.index).fillna(0).astype(int)

    model = F1Predictor()
    model.train(X_train, y_train, y_dnf=train_dnf)

    if X_test.empty:
        logger.warning("No test data available — skipping evaluation")
        metrics = {"position_mae": None, "podium_accuracy": None, "winner_accuracy": None}
    else:
        metrics = model.evaluate(X_test, y_test, y_dnf=test_dnf)

    return model, metrics
