"""
Hyperparameter Optimization for XGBoost Module.

This module provides tools for automated hyperparameter tuning using Optuna, including:
- HyperparameterTuner: Automated hyperparameter search with Optuna
- StudyManager: Management and persistence of optimization studies
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import pandas as pd
from optuna import Study
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from config import (
    MODEL_FEATURES,
    TARGET_HORIZON,
    TARGET_LABELS,
    XGBOOST_PARAMS,
)
from modules.common.utils import log_info, log_success, log_warn
from modules.xgboost.model import _resolve_xgb_classifier


class StudyManager:
    """
    Manage and persist results of optimization studies.

    Stores study metadata, including:
    - Best parameters
    - Best score
    - Timestamp
    - Symbol and timeframe
    - Study history
    """

    def __init__(self, storage_dir: str = "artifacts/xgboost/optimization"):
        """
        Initialize StudyManager.

        Args:
            storage_dir: Directory to store studies (relative to project root)
        """
        # [DEBUG] Path resolution checked - resolves from current working directory
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
        Save study metadata to a JSON file.

        Args:
            study: Optuna Study object
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (e.g., "1h")
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

        # Store trial history (save only completed trials)
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
        Load best parameters from the most recent study.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            max_age_days: Maximum age (in days) allowed for a cached study (default: 30)

        Returns:
            Best parameters dict, or None if not found
        """
        pattern = f"study_{symbol}_{timeframe}_*.json"
        study_files = sorted(self.storage_dir.glob(pattern), reverse=True)

        if not study_files:
            # [DEBUG] No study files found - checked pattern matching
            return None

        # Load the most recent study
        latest_study = study_files[0]
        # [DEBUG] Study file loading - error handling added for JSON parsing and timestamp validation
        try:
            with open(latest_study, "r", encoding="utf-8") as f:
                study_data = json.load(f)
        except Exception:
            return None

        # Check age of study
        try:
            study_timestamp = datetime.strptime(study_data["timestamp"], "%Y%m%d_%H%M%S")
            age_days = (datetime.now() - study_timestamp).days
        except (KeyError, ValueError):
            # [DEBUG] Timestamp parsing error handling added
            return None

        if age_days > max_age_days:
            # [DEBUG] Study age validation checked
            log_warn(f"Study is {age_days} days old (max: {max_age_days}). Consider re-running optimization.")
            return None

        log_info(f"Loaded best params from study ({age_days} days old): score={study_data['best_score']:.4f}")
        return study_data["best_params"]


class HyperparameterTuner:
    """
    Integrates Optuna to search for the best parameter set for XGBoost.

    Uses TimeSeriesSplit cross-validation with gap prevention
    to avoid data leakage.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        storage_dir: str = "artifacts/xgboost/optimization",
    ):
        """
        Initialize HyperparameterTuner.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (e.g., "1h")
            storage_dir: Directory to store studies
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.study_manager = StudyManager(storage_dir)
        self.classifier_cls = _resolve_xgb_classifier()

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
        # Define search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "n_jobs": -1,
            "num_class": len(TARGET_LABELS),
        }

        # Filter parameters through whitelist if classifier has one
        whitelist = getattr(self.classifier_cls, "XGB_PARAM_WHITELIST", None)
        if whitelist is not None:
            params = {k: v for k, v in params.items() if k in whitelist}

        # Time-Series Cross-Validation with gap prevention
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for train_idx, test_idx in tscv.split(X):
            # [DEBUG] CV fold processing - gap prevention and class diversity validation checked
            # Apply gap to prevent data leakage
            train_idx_array = np.array(train_idx)
            if len(train_idx_array) > TARGET_HORIZON:
                train_idx_filtered = train_idx_array[:-TARGET_HORIZON]
            else:
                continue

            # Ensure test set doesn't overlap with gap
            # [DEBUG] test_idx_array scope and gap filtering logic verified
            test_idx_array = np.array(test_idx)
            if len(train_idx_filtered) > 0 and len(test_idx_array) > 0:
                min_test_start = train_idx_filtered[-1] + TARGET_HORIZON + 1
                if test_idx_array[0] < min_test_start:
                    test_idx_array = test_idx_array[test_idx_array >= min_test_start]
                    if len(test_idx_array) == 0:
                        continue

            # Class diversity validation
            y_train_fold = y.iloc[train_idx_filtered]
            unique_classes = sorted(y_train_fold.unique())

            if len(unique_classes) < len(TARGET_LABELS):
                continue

            # Train model with trial parameters
            model = self.classifier_cls(**params)
            model.fit(X.iloc[train_idx_filtered], y.iloc[train_idx_filtered])

            # Evaluate on test set
            # [DEBUG] test_idx_array usage verified - correctly used after filtering
            if len(test_idx_array) > 0:
                y_test_fold = y.iloc[test_idx_array]
                preds = model.predict(X.iloc[test_idx_array])
                acc = accuracy_score(y_test_fold, preds)
                cv_scores.append(acc)

        if len(cv_scores) == 0:
            # Return low score if no valid folds
            return 0.0

        return float(np.mean(cv_scores))

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
            df: DataFrame containing features (MODEL_FEATURES) and target ("Target")
            n_trials: Number of trials to run
            n_splits: Number of folds for cross-validation
            study_name: Name for the study (default: auto-generated)
            load_existing: Whether to load an existing study

        Returns:
            Dictionary containing best parameters
        """
        # [DEBUG] Input validation - MODEL_FEATURES and Target column validation added
        try:
            X = df[MODEL_FEATURES]
        except KeyError:
            raise
        # Validate Target column
        if "Target" not in df.columns:
            raise ValueError("DataFrame must contain 'Target' column")
        try:
            y = df["Target"].astype(int)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert 'Target' column to int: {e}") from e

        # Check data length
        if len(df) < 100:
            log_warn(f"Insufficient data for optimization (need >= 100, got {len(df)})")
            return XGBOOST_PARAMS.copy()

        # Create study name if not provided
        if study_name is None:
            study_name = f"xgboost_{self.symbol}_{self.timeframe}"

        # Create storage path for Optuna (absolute path from artifacts)
        storage_path = self.study_manager.storage_dir / "studies.db"
        # Use absolute path to ensure Optuna finds the correct file
        storage_url = f"sqlite:///{storage_path.resolve()}"

        # Load existing study if available
        # [DEBUG] Exception handling improved - added DuplicatedStudyError handling for Optuna
        study = None
        if load_existing:
            try:
                study = optuna.load_study(
                    study_name=study_name,
                    storage=storage_url,
                )
                log_info(f"Loaded existing study: {study_name}")
            except (ValueError, KeyError):
                # Study does not exist, will create new
                pass
            except optuna.exceptions.DuplicatedStudyError:
                # Study exists but conflict, create new with load_if_exists=True
                pass
            except Exception:
                # Study does not exist or other error, will create new
                pass

        # Create new study if not loaded
        # [DEBUG] DuplicatedStudyError handling with fallback to load_study added
        if study is None:
            try:
                study = optuna.create_study(
                    study_name=study_name,
                    direction="maximize",
                    storage=storage_url,
                    load_if_exists=True,
                )
                log_info(f"Created new study: {study_name}")
            except optuna.exceptions.DuplicatedStudyError as e:
                # Study exists, load it
                try:
                    study = optuna.load_study(
                        study_name=study_name,
                        storage=storage_url,
                    )
                    log_info(f"Loaded existing study after conflict: {study_name}")
                except Exception as load_e:
                    raise RuntimeError(f"Failed to create or load study: {e}") from load_e
            except Exception:
                raise

        # Run optimization
        log_info(f"Starting optimization with {n_trials} trials...")
        study.optimize(
            lambda trial: self._objective(trial, X, y, n_splits=n_splits),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        # Get best parameters
        best_params = study.best_params.copy()
        best_score = study.best_value

        # Add fixed parameters that are not optimized
        best_params.update(
            {
                "random_state": 42,
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "n_jobs": -1,
                "num_class": len(TARGET_LABELS),
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
            df: DataFrame containing features and target
            n_trials: Number of trials to run if optimizing anew
            use_cached: Whether to use cached params

        Returns:
            Dictionary containing best parameters
        """
        # Try loading cached params
        if use_cached:
            cached_params = self.study_manager.load_best_params(
                symbol=self.symbol, timeframe=self.timeframe, max_age_days=30
            )
            if cached_params is not None:
                log_info("Using cached best parameters")
                return cached_params

        # Run fresh optimization
        log_info("No cached parameters found, running optimization...")
        return self.optimize(df, n_trials=n_trials)
