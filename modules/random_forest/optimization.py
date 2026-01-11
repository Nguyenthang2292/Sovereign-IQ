
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

from config import (
from optuna import Study
from optuna.exceptions import DuplicatedStudyError
import optuna
from optuna.exceptions import DuplicatedStudyError
import optuna

"""
Hyperparameter Optimization for Random Forest Module.

This module provides tools for automated hyperparameter tuning using Optuna,
including:
- HyperparameterTuner: Automated hyperparameter search with Optuna
- StudyManager: Management and persistence of optimization studies
"""



    MIN_TRAINING_SAMPLES,
    MODEL_RANDOM_STATE,
)
from modules.common.ui.logging import log_error, log_info, log_success, log_warn
from modules.random_forest.utils.data_preparation import prepare_training_data
from modules.random_forest.utils.training import apply_smote


class StudyManager:
    """
    Manage and store results of optimization studies.

    Stores study metadata including:
    - Best parameters
    - Best score
    - Timestamp
    - Symbol and timeframe
    - Study history
    """

    def __init__(self, storage_dir: str = "artifacts/random_forest/optimization"):
        """
        Initialize StudyManager.

        Args:
            storage_dir: Directory to store studies (relative to project root)
        """
        # Resolve path from project root to ensure absolute path
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_study(
        self,
        study: Study,
        symbol: str,
        timeframe: str,
        best_params: Dict[str, Any],
        best_score: float,
    ) -> str:
        """
        Save study metadata to JSON file.

        Args:
            study: Optuna Study object
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h')
            best_params: Best hyperparameters found
            best_score: Best score achieved

        Returns:
            Path to saved study file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"study_{symbol}_{timeframe}_{timestamp}.json"
        filepath = self.storage_dir / filename

        study_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": timestamp,
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": len(study.trials),
            "study_name": study.study_name,
            "direction": study.direction.name,
        }

        # Save trial history (only completed trials)
        study_data["trials"] = [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            }
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(study_data, f, indent=2, default=str)

        log_success(f"Study saved to: {filepath}")
        return str(filepath)

    def load_best_params(self, symbol: str, timeframe: str, max_age_days: int = 30) -> Optional[Dict[str, Any]]:
        """
        Load best parameters from latest study.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            max_age_days: Maximum age of study in days (default: 30)

        Returns:
            Best parameters dict or None if not found
        """
        pattern = f"study_{symbol}_{timeframe}_*.json"
        study_files = sorted(self.storage_dir.glob(pattern), reverse=True)

        if not study_files:
            return None

        # Load latest study
        latest_study = study_files[0]
        try:
            with open(latest_study, "r", encoding="utf-8") as f:
                study_data = json.load(f)
        except Exception as e:
            log_error(f"Failed to read study JSON file '{latest_study}': {type(e).__name__}: {str(e)}")
            return None

        # Check age of study
        try:
            study_timestamp = datetime.strptime(study_data["timestamp"], "%Y%m%d_%H%M%S")
            age_days = (datetime.now() - study_timestamp).days
        except (KeyError, ValueError) as e:
            timestamp_value = study_data.get("timestamp", "missing")
            log_error(
                f"Failed to parse timestamp from study file '{latest_study}': "
                f"{type(e).__name__}: {str(e)}. "
                f"Timestamp value: {repr(timestamp_value)}"
            )
            return None

        if age_days > max_age_days:
            log_warn(f"Study is {age_days} days old (max: {max_age_days}). Consider re-running optimization.")
            return None

        log_info(f"Loaded best params from study ({age_days} days old): score={study_data['best_score']:.4f}")
        return study_data["best_params"]


class HyperparameterTuner:
    """
    Integrate Optuna to find the best parameters for Random Forest.

    Use TimeSeriesSplit cross-validation with gap prevention
    to ensure no data leakage.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        storage_dir: str = "artifacts/random_forest/optimization",
    ):
        """
        Initialize HyperparameterTuner.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h')
            storage_dir: Directory to store studies (relative to project root)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.study_manager = StudyManager(storage_dir)
        # Target horizon for gap prevention (5 periods for Random Forest)
        self.target_horizon = 5

    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of folds for cross-validation

        Returns:
            Mean CV accuracy score
        """
        # Define search space for Random Forest
        # For max_depth, use suggest_int with None as a special case
        # We'll suggest a value 0-30, where 0 means None
        max_depth_val = trial.suggest_int("max_depth", 0, 30)
        max_depth = None if max_depth_val == 0 else max_depth_val

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": max_depth,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": MODEL_RANDOM_STATE,
            "n_jobs": -1,
        }

        # Time-Series Cross-Validation with gap prevention
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        skipped_folds = 0
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Apply gap to prevent data leakage (target horizon = 5 periods)
            train_idx_array = np.array(train_idx)
            if len(train_idx_array) <= self.target_horizon:
                skipped_folds += 1
                log_warn(
                    f"Fold {fold_idx + 1}: Training set too small ({len(train_idx_array)} <= {self.target_horizon}), skipping"
                )
                continue

            train_idx_filtered = train_idx_array[: -self.target_horizon]

            # Ensure test set doesn't overlap with gap
            test_idx_array = np.array(test_idx)
            if len(train_idx_filtered) == 0 or len(test_idx_array) == 0:
                skipped_folds += 1
                log_warn(f"Fold {fold_idx + 1}: Empty train or test set after gap prevention, skipping")
                continue

            min_test_start = train_idx_filtered[-1] + self.target_horizon + 1
            if test_idx_array[0] < min_test_start:
                test_idx_array = test_idx_array[test_idx_array >= min_test_start]
                if len(test_idx_array) == 0:
                    skipped_folds += 1
                    log_warn(f"Fold {fold_idx + 1}: Test set empty after gap adjustment, skipping")
                    continue

            # Get data for this fold
            X_train_fold = X.iloc[train_idx_filtered]
            y_train_fold = y.iloc[train_idx_filtered]

            # Check class diversity
            unique_classes = sorted(y_train_fold.unique())
            if len(unique_classes) < 2:  # Need at least 2 classes
                skipped_folds += 1
                log_warn(
                    f"Fold {fold_idx + 1}: Insufficient class diversity ({len(unique_classes)} class(es)), skipping"
                )
                continue

            # Apply SMOTE to training fold
            try:
                X_train_resampled, y_train_resampled = apply_smote(X_train_fold, y_train_fold)
            except (ValueError, RuntimeError, MemoryError):
                # If SMOTE fails due to known issues (e.g. no SMOTE available), use original data
                X_train_resampled, y_train_resampled = X_train_fold, y_train_fold
            except Exception as e:
                # Unexpected SMOTE error - log full error and re-raise
                log_error(f"Unexpected SMOTE error in fold {fold_idx + 1}: {type(e).__name__}: {str(e)}")
                raise

            # Compute class weights (balanced weights for imbalanced classes)
            class_weights = compute_class_weight("balanced", classes=np.unique(y_train_resampled), y=y_train_resampled)
            weight_dict = {int(cls): weight for cls, weight in zip(np.unique(y_train_resampled), class_weights)}

            # Create model with optimized params + class weights
            model_params = params.copy()
            model_params["class_weight"] = weight_dict
            model = RandomForestClassifier(**model_params)

            # Train model
            try:
                model.fit(X_train_resampled, y_train_resampled)
            except (ValueError, RuntimeError) as e:
                # Known training errors - skip this fold
                log_warn(f"Model training failed in fold: {e}")
                continue
            except Exception as e:
                # Unexpected training error - log full error and re-raise
                log_error(f"Unexpected training error in fold {fold_idx + 1}: {type(e).__name__}: {str(e)}")
                raise

            # Evaluate on test set
            if len(test_idx_array) > 0:
                X_test_fold = X.iloc[test_idx_array]
                y_test_fold = y.iloc[test_idx_array]
                try:
                    preds = model.predict(X_test_fold)
                    acc = accuracy_score(y_test_fold, preds)
                    cv_scores.append(acc)
                except (ValueError, RuntimeError) as e:
                    # Known prediction errors - skip this fold (should never happen)
                    log_warn(f"Prediction failed in fold: {e}")
                    continue
                except Exception as e:
                    # Unexpected prediction error - log full error and re-raise
                    log_error(f"Unexpected prediction error in fold {fold_idx + 1}: {type(e).__name__}: {str(e)}")
                    raise

        if len(cv_scores) == 0:
            # Return low score if no valid folds (should never happen)
            log_warn(
                f"No valid CV folds after gap prevention and filtering. "
                f"Skipped {skipped_folds}/{n_splits} folds. Returning 0.0"
            )
            return 0.0

        mean_score = float(np.mean(cv_scores))
        log_info(
            f"CV completed with {len(cv_scores)}/{n_splits} valid folds "
            f"({skipped_folds} skipped). Mean accuracy: {mean_score:.4f}"
        )
        return mean_score

    def _load_or_create_study(self, study_name: str, storage_url: str, load_existing: bool) -> Study:
        """
        Load existing study or create new one, handling all exception cases.

        Args:
            study_name: Name of the study
            storage_url: SQLite storage URL for Optuna
            load_existing: Whether to attempt loading existing study first

        Returns:
            Optuna Study object

        Raises:
            optuna.exceptions.StorageInternalError: On storage errors when loading
            Exception: On unexpected errors when loading
            RuntimeError: On failure to create or load study after conflict
        """
        study = None

        # Load existing study if requested
        if load_existing:
            try:
                study = optuna.load_study(
                    study_name=study_name,
                    storage=storage_url,
                )
                log_info(f"Loaded existing study: {study_name}")
            except (ValueError, KeyError):
                # Study does not exist, create new one
                log_info(f"Study '{study_name}' not found, will create new study")
            except optuna.exceptions.StorageInternalError as e:
                log_error(f"Storage error when loading study: {e}")
                raise
            except Exception as e:
                log_error(f"Unexpected error when loading study: {e}")
                raise

        # Create new study if not exists
        if study is None:
            try:
                study = optuna.create_study(
                    study_name=study_name,
                    direction="maximize",
                    storage=storage_url,
                    load_if_exists=False,
                )
                log_info(f"Created new study: {study_name}")
            except DuplicatedStudyError:
                # Study already exists, load it
                study = optuna.load_study(
                    study_name=study_name,
                    storage=storage_url,
                )
                log_info(f"Loaded existing study: {study_name}")

        return study

    def optimize(
        self,
        df: pd.DataFrame,
        n_trials: int = 100,
        n_splits: int = 5,
        study_name: Optional[str] = None,
        load_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            df: DataFrame containing OHLCV data (will be prepared by prepare_training_data)
            n_trials: Number of trials to run
            n_splits: Number of folds for cross-validation
            study_name: Name of study (default: auto-generated)
            load_existing: Load existing study if True

        Returns:
            Dictionary containing best parameters
        """
        # Input validation
        if df is None:
            raise ValueError("Input DataFrame is None")  # pyright: ignore[reportUnreachable]
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input must be a pandas DataFrame, got {type(df)}")  # pyright: ignore[reportUnreachable]
        if n_trials <= 0:
            raise ValueError(f"n_trials must be positive, got {n_trials}")
        if n_splits <= 0:
            raise ValueError(f"n_splits must be positive, got {n_splits}")

        # Prepare training data
        prepared_data = prepare_training_data(df)
        if prepared_data is None:
            raise ValueError("Failed to prepare training data")

        features, target = prepared_data

        # Validate data
        if len(features) < MIN_TRAINING_SAMPLES:
            log_warn(f"Insufficient data for optimization (need >= {MIN_TRAINING_SAMPLES}, got {len(features)})")
            # Return default params (should never happen)
            return {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "random_state": MODEL_RANDOM_STATE,
                "n_jobs": -1,
            }

        # Check class diversity
        unique_classes = sorted(target.unique())
        if len(unique_classes) < 2:
            raise ValueError("Need at least 2 classes for optimization")

        # Create study name if not provided
        if study_name is None:
            study_name = f"random_forest_{self.symbol}_{self.timeframe}"

        # Create storage path for Optuna (absolute path from artifacts)
        storage_path = self.study_manager.storage_dir / "studies.db"
        storage_url = f"sqlite:///{storage_path.resolve()}"

        # Load existing study or create new one
        study = self._load_or_create_study(study_name, storage_url, load_existing)

        # Run optimization
        log_info(f"Starting optimization with {n_trials} trials...")
        study.optimize(
            lambda trial: self._objective(trial, features, target, n_splits=n_splits),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        # Get best parameters
        best_params = study.best_params.copy()
        best_score = study.best_value

        # Reconstruct max_depth (0 means None)
        if "max_depth" in best_params:
            max_depth_val = best_params["max_depth"]
            if max_depth_val == 0:
                best_params["max_depth"] = None
            # Otherwise keep the value as is

        # Add fixed parameters that were not optimized
        best_params.update(
            {
                "random_state": MODEL_RANDOM_STATE,
                "n_jobs": -1,
            }
        )

        log_success(f"Optimization completed! Best CV accuracy: {best_score:.4f}\nBest parameters: {best_params}")

        # Save study
        self.study_manager.save_study(
            study=study,
            symbol=self.symbol,
            timeframe=self.timeframe,
            best_params=best_params,
            best_score=best_score,
        )

        return best_params

    def get_best_params(self, df: pd.DataFrame, n_trials: int = 100, use_cached: bool = True) -> Dict[str, Any]:
        """
        Get best parameters, using cached if available.

        Args:
            df: DataFrame containing OHLCV data
            n_trials: Number of trials to run if new optimization is needed
            use_cached: Use cached params if True

        Returns:
            Dictionary containing best parameters
        """
        # Try load cached params
        if use_cached:
            cached_params = self.study_manager.load_best_params(
                symbol=self.symbol, timeframe=self.timeframe, max_age_days=30
            )
            if cached_params is not None:
                log_info("Using cached best parameters")
                return cached_params

        # Run new optimization
        log_info("No cached parameters found, running optimization...")
        return self.optimize(df, n_trials=n_trials)
