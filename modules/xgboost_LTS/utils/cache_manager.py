"""
Cache manager for XGBoost models and data.
Uses hashing to ensure cache validity.
Saves/loads models in XGBoost native format (.json) for cross-version compatibility;
falls back to .joblib for legacy caches.
"""

import hashlib
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import xgboost as xgb

from config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

# Native format avoids pickle/joblib compatibility warnings across XGBoost versions
NATIVE_EXT = ".json"
LEGACY_EXT = ".joblib"


class CacheManager:
    """Manage caching of XGBoost models and labeled datasets."""

    def __init__(self, subsystem: str = "xgboost"):
        """
        Initialize cache manager.

        Args:
            subsystem: Subsystem name (e.g., 'xgboost')
        """
        self.cache_dir = Path(ARTIFACTS_DIR) / subsystem
        self.models_dir = self.cache_dir / "models"
        self.labels_dir = self.cache_dir / "labels"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def _compute_df_hash(self, df: pd.DataFrame) -> str:
        """
        Compute hash of DataFrame content.

        Uses pandas built-in hashing utility for efficiency.

        Args:
            df: Input DataFrame to hash

        Returns:
            Hash string (first 16 chars of SHA256)
        """
        return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]

    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Compute hash of configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Hash string (first 16 chars of SHA256)
        """
        # Sort keys for deterministic JSON
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]

    def get_model_path(
        self, df: pd.DataFrame, config: Dict[str, Any], suffix: str = "", native: bool = True
    ) -> Path:
        """
        Generate unique cache path for a model.

        Args:
            df: Training DataFrame
            config: Model configuration
            suffix: Optional filename suffix
            native: If True, return path with .json (XGBoost native); else .joblib (legacy)

        Returns:
            Path to cached model file
        """
        df_hash = self._compute_df_hash(df)
        config_hash = self._compute_config_hash(config)
        ext = NATIVE_EXT if native else LEGACY_EXT
        filename = f"model_{df_hash}_{config_hash}{suffix}{ext}"
        return self.models_dir / filename

    def load_model(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[Any]:
        """
        Load model from cache if exists.
        Prefers native .json format; falls back to legacy .joblib.

        Args:
            df: Training DataFrame
            config: Model configuration

        Returns:
            Loaded model if cache hit, None otherwise
        """
        path_native = self.get_model_path(df, config, native=True)
        path_joblib = self.get_model_path(df, config, native=False)

        if path_native.exists():
            try:
                logger.info(f"Loading cached model from {path_native.name} (native format)")
                model = xgb.XGBClassifier()
                model.load_model(str(path_native))
                return model
            except Exception as e:
                logger.warning(f"Failed to load native cached model: {e}")

        if path_joblib.exists():
            try:
                logger.info(f"Loading cached model from {path_joblib.name} (legacy joblib)")
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*serialized model.*older version.*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=".*Booster.save_model.*",
                        category=UserWarning,
                    )
                return joblib.load(path_joblib)
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}")
        return None

    def save_model(self, model: Any, df: pd.DataFrame, config: Dict[str, Any]):
        """
        Save model to cache in XGBoost native format (.json).
        Uses Booster.save_model() for cross-version compatibility.

        Args:
            model: Trained XGBoost model (XGBClassifier or compatible)
            df: Training DataFrame
            config: Model configuration
        """
        path = self.get_model_path(df, config, native=True)
        try:
            if hasattr(model, "save_model"):
                model.save_model(str(path))
                logger.info(f"Saved model to cache: {path.name} (native format)")
            else:
                path_joblib = self.get_model_path(df, config, native=False)
                joblib.dump(model, path_joblib, compress=3)
                logger.info(f"Saved model to cache: {path_joblib.name} (legacy joblib)")
        except Exception as e:
            logger.error(f"Failed to save model to cache: {e}")

    def get_labels_path(self, df: pd.DataFrame, labeling_config: Dict[str, Any]) -> Path:
        """
        Generate unique cache path for labeled data.

        Args:
            df: Source DataFrame (before labeling)
            labeling_config: Labeling configuration

        Returns:
            Path to cached labels file
        """
        # Hash only columns used for labeling (OHLCV)
        # to properly detect data changes
        cols_to_hash = ["close", "high", "low", "volume", "open"]
        # Filter existing columns
        cols = [c for c in cols_to_hash if c in df.columns]

        if cols:
            df_hash = self._compute_df_hash(df[cols])
        else:
            # Fallback to full df hash if no OHLCV columns
            df_hash = self._compute_df_hash(df)

        config_hash = self._compute_config_hash(labeling_config)
        filename = f"labels_{df_hash}_{config_hash}.parquet"
        return self.labels_dir / filename

    def load_labels(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Load labeled DataFrame from cache if exists.

        Args:
            df: Source DataFrame (before labeling)
            config: Labeling configuration

        Returns:
            Cached labeled DataFrame if cache hit, None otherwise
        """
        path = self.get_labels_path(df, config)
        if path.exists():
            try:
                logger.info(f"Loading cached labels from {path.name}")
                return pd.read_parquet(path)
            except Exception as e:
                logger.warning(f"Failed to load cached labels: {e}")
        return None

    def save_labels(self, labeled_df: pd.DataFrame, source_df: pd.DataFrame, config: Dict[str, Any]):
        """
        Save labeled DataFrame to cache.

        Args:
            labeled_df: DataFrame with labels
            source_df: Source DataFrame (before labeling)
            config: Labeling configuration
        """
        path = self.get_labels_path(source_df, config)
        try:
            labeled_df.to_parquet(path, compression="snappy")
            logger.info(f"Saved labels to cache: {path.name}")
        except Exception as e:
            logger.error(f"Failed to save labels to cache: {e}")

    def clear_cache(self) -> None:
        """Remove all cached model and label files in this manager's cache directory."""
        for d in (self.models_dir, self.labels_dir):
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        try:
                            f.unlink()
                            logger.info(f"Removed cache file: {f.name}")
                        except OSError as e:
                            logger.warning(f"Failed to remove {f}: {e}")
