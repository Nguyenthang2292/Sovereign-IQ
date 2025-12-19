"""
Hyperparameter Optimization for Random Forest Module.

This module provides tools for automated hyperparameter tuning using Optuna,
including:
- HyperparameterTuner: Automated hyperparameter search with Optuna
- StudyManager: Management and persistence of optimization studies
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna import Study
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from config import (
    MODEL_FEATURES,
    MODEL_RANDOM_STATE,
    MIN_TRAINING_SAMPLES,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
)
from modules.common.ui.logging import log_info, log_model, log_success, log_warn
from modules.random_forest.utils.data_preparation import prepare_training_data
from modules.random_forest.utils.training import apply_smote, create_model_and_weights


class StudyManager:
    """
    Quản lý và lưu trữ kết quả optimization studies.
    
    Lưu trữ metadata của study bao gồm:
    - Best parameters
    - Best score
    - Timestamp
    - Symbol và timeframe
    - Study history
    """

    def __init__(self, storage_dir: str = "artifacts/random_forest/optimization"):
        """
        Khởi tạo StudyManager.
        
        Args:
            storage_dir: Thư mục lưu trữ studies (relative từ project root)
        """
        # Resolve path từ project root để đảm bảo absolute path
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
        Lưu study metadata vào file JSON.
        
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

        # Lưu trial history (chỉ lưu completed trials)
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

    def load_best_params(
        self, symbol: str, timeframe: str, max_age_days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Load best parameters từ study gần nhất.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            max_age_days: Số ngày tối đa cho phép study cũ (default: 30)
            
        Returns:
            Best parameters dict hoặc None nếu không tìm thấy
        """
        pattern = f"study_{symbol}_{timeframe}_*.json"
        study_files = sorted(self.storage_dir.glob(pattern), reverse=True)

        if not study_files:
            return None

        # Load study mới nhất
        latest_study = study_files[0]
        try:
            with open(latest_study, "r", encoding="utf-8") as f:
                study_data = json.load(f)
        except Exception as e:
            return None

        # Kiểm tra tuổi của study
        try:
            study_timestamp = datetime.strptime(study_data["timestamp"], "%Y%m%d_%H%M%S")
            age_days = (datetime.now() - study_timestamp).days
        except (KeyError, ValueError) as e:
            return None

        if age_days > max_age_days:
            log_warn(
                f"Study is {age_days} days old (max: {max_age_days}). "
                "Consider re-running optimization."
            )
            return None

        log_info(
            f"Loaded best params from study ({age_days} days old): "
            f"score={study_data['best_score']:.4f}"
        )
        return study_data["best_params"]


class HyperparameterTuner:
    """
    Tích hợp Optuna để tìm kiếm bộ tham số tốt nhất cho Random Forest.
    
    Sử dụng TimeSeriesSplit cross-validation với gap prevention
    để đảm bảo không có data leakage.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        storage_dir: str = "artifacts/random_forest/optimization",
    ):
        """
        Khởi tạo HyperparameterTuner.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (e.g., "1h")
            storage_dir: Thư mục lưu trữ studies
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.study_manager = StudyManager(storage_dir)
        # Target horizon for gap prevention (5 periods for RF)
        self.target_horizon = 5

    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> float:
        """
        Objective function cho Optuna optimization.
        
        Args:
            trial: Optuna trial object
            X: Feature DataFrame
            y: Target Series
            n_splits: Số folds cho cross-validation
            
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

        # Time-Series Cross-Validation với gap prevention
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        skipped_folds = 0
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Apply gap to prevent data leakage (target horizon = 5)
            train_idx_array = np.array(train_idx)
            if len(train_idx_array) <= self.target_horizon:
                skipped_folds += 1
                log_warn(f"Fold {fold_idx + 1}: Training set too small ({len(train_idx_array)} <= {self.target_horizon}), skipping")
                continue
            
            train_idx_filtered = train_idx_array[:-self.target_horizon]

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
                log_warn(f"Fold {fold_idx + 1}: Insufficient class diversity ({len(unique_classes)} class(es)), skipping")
                continue

            # Apply SMOTE to training fold
            try:
                X_train_resampled, y_train_resampled = apply_smote(X_train_fold, y_train_fold)
            except (ValueError, RuntimeError, MemoryError) as e:
                # If SMOTE fails due to known issues, use original data
                X_train_resampled, y_train_resampled = X_train_fold, y_train_fold
            except Exception as e:
                # Unexpected SMOTE error - log and skip this fold
                log_warn(f"Unexpected SMOTE error in fold: {e}")
                continue

            # Compute class weights
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train_resampled),
                y=y_train_resampled
            )
            weight_dict = {
                int(cls): weight
                for cls, weight in zip(np.unique(y_train_resampled), class_weights)
            }

            # Create model with optimized params + class weights
            model_params = params.copy()
            model_params['class_weight'] = weight_dict
            model = RandomForestClassifier(**model_params)

            # Train model
            try:
                model.fit(X_train_resampled, y_train_resampled)
            except (ValueError, RuntimeError) as e:
                # Known training errors - skip this fold
                log_warn(f"Model training failed in fold: {e}")
                continue
            except Exception as e:
                # Unexpected training error - log and skip
                log_warn(f"Unexpected training error in fold: {e}")
                continue

            # Evaluate on test set
            if len(test_idx_array) > 0:
                X_test_fold = X.iloc[test_idx_array]
                y_test_fold = y.iloc[test_idx_array]
                try:
                    preds = model.predict(X_test_fold)
                    acc = accuracy_score(y_test_fold, preds)
                    cv_scores.append(acc)
                except (ValueError, RuntimeError) as e:
                    # Known prediction errors - skip this fold
                    log_warn(f"Prediction failed in fold: {e}")
                    continue
                except Exception as e:
                    # Unexpected prediction error - log and skip
                    log_warn(f"Unexpected prediction error in fold: {e}")
                    continue

        if len(cv_scores) == 0:
            # Return low score if no valid folds
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

    def optimize(
        self,
        df: pd.DataFrame,
        n_trials: int = 100,
        n_splits: int = 5,
        study_name: Optional[str] = None,
        load_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Chạy hyperparameter optimization.
        
        Args:
            df: DataFrame chứa OHLCV data (sẽ được prepare_training_data)
            n_trials: Số lượng trials để chạy
            n_splits: Số folds cho cross-validation
            study_name: Tên study (mặc định: auto-generated)
            load_existing: Có load study đã tồn tại không
            
        Returns:
            Dictionary chứa best parameters
        """
        # Input validation
        if df is None:
            raise ValueError("Input DataFrame is None")
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input must be a pandas DataFrame, got {type(df)}")
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
            log_warn(
                f"Insufficient data for optimization (need >= {MIN_TRAINING_SAMPLES}, got {len(features)})"
            )
            # Return default params
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

        # Tạo study name nếu chưa có
        if study_name is None:
            study_name = f"random_forest_{self.symbol}_{self.timeframe}"

        # Tạo storage path cho Optuna (absolute path từ artifacts)
        storage_path = self.study_manager.storage_dir / "studies.db"
        storage_url = f"sqlite:///{storage_path.resolve()}"

        # Load existing study nếu có
        study = None
        if load_existing:
            try:
                study = optuna.load_study(
                    study_name=study_name,
                    storage=storage_url,
                )
                log_info(f"Loaded existing study: {study_name}")
            except (ValueError, KeyError) as e:
                # Study chưa tồn tại, tạo mới
                log_info(f"Study '{study_name}' not found, will create new study")
            except optuna.exceptions.DuplicatedStudyError as e:
                # Study đã tồn tại nhưng có conflict
                log_warn(f"Study conflict detected: {e}. Will attempt to load or create new study.")
            except optuna.exceptions.StorageInternalError as e:
                log_error(f"Storage error when loading study: {e}")
                raise
            except Exception as e:
                log_error(f"Unexpected error when loading study: {e}")
                raise

        # Tạo study mới nếu chưa có
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
                try:
                    study = optuna.load_study(
                        study_name=study_name,
                        storage=storage_url,
                    )
                    log_info(f"Loaded existing study after conflict: {study_name}")
                except Exception as load_e:
                    raise RuntimeError(f"Failed to create or load study: {e}") from load_e
            except Exception as e:
                raise

        # Chạy optimization
        log_info(f"Starting optimization with {n_trials} trials...")
        study.optimize(
            lambda trial: self._objective(trial, features, target, n_splits=n_splits),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        # Lấy best parameters
        best_params = study.best_params.copy()
        best_score = study.best_value

        # Reconstruct max_depth (0 means None)
        if "max_depth" in best_params:
            max_depth_val = best_params["max_depth"]
            if max_depth_val == 0:
                best_params["max_depth"] = None
            # Otherwise keep the value as is

        # Thêm các parameters cố định không được optimize
        best_params.update({
            "random_state": MODEL_RANDOM_STATE,
            "n_jobs": -1,
        })

        log_success(
            f"Optimization completed! Best CV accuracy: {best_score:.4f}\n"
            f"Best parameters: {best_params}"
        )

        # Lưu study
        self.study_manager.save_study(
            study=study,
            symbol=self.symbol,
            timeframe=self.timeframe,
            best_params=best_params,
            best_score=best_score,
        )

        return best_params

    def get_best_params(
        self, df: pd.DataFrame, n_trials: int = 100, use_cached: bool = True
    ) -> Dict[str, Any]:
        """
        Lấy best parameters, sử dụng cached nếu có.
        
        Args:
            df: DataFrame chứa OHLCV data
            n_trials: Số trials nếu cần optimize mới
            use_cached: Có sử dụng cached params không
            
        Returns:
            Dictionary chứa best parameters
        """
        # Thử load cached params
        if use_cached:
            cached_params = self.study_manager.load_best_params(
                symbol=self.symbol, timeframe=self.timeframe, max_age_days=30
            )
            if cached_params is not None:
                log_info("Using cached best parameters")
                return cached_params

        # Chạy optimization mới
        log_info("No cached parameters found, running optimization...")
        return self.optimize(df, n_trials=n_trials)

