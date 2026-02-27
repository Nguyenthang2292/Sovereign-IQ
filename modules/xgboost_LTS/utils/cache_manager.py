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
import numpy as np
import pandas as pd
import xgboost as xgb

from config import ARTIFACTS_DIR
from modules.xgboost_LTS.utils.memory_map import dataframe_to_memmap

logger = logging.getLogger(__name__)

# Native format avoids pickle/joblib compatibility warnings across XGBoost versions
NATIVE_EXT = ".json"
LEGACY_EXT = ".joblib"


class CacheManager:
    """Manage caching of XGBoost models and labeled datasets."""

    def __init__(self, subsystem: str = "xgboost", max_cache_entries: Optional[int] = None):
        """
        Initialize cache manager.

        Args:
            subsystem: Subsystem name (e.g., 'xgboost')
            max_cache_entries: Optional max files per cache subdirectory.
                When exceeded, oldest files are evicted.
        """
        self.cache_dir = Path(ARTIFACTS_DIR) / subsystem
        self.models_dir = self.cache_dir / "models"
        self.labels_dir = self.cache_dir / "labels"
        self.max_cache_entries = max_cache_entries

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def _evict_oldest(self, directory: Path) -> None:
        """Evict oldest files in directory when exceeding max_cache_entries."""
        if self.max_cache_entries is None or self.max_cache_entries <= 0:
            return

        files = [f for f in directory.iterdir() if f.is_file()]
        if len(files) <= self.max_cache_entries:
            return

        files.sort(key=lambda file_path: file_path.stat().st_mtime)
        excess_count = len(files) - self.max_cache_entries
        for file_path in files[:excess_count]:
            try:
                file_path.unlink()
                logger.info(f"Evicted old cache file: {file_path.name}")
            except OSError as e:
                logger.warning(f"Failed to evict cache file {file_path}: {e}")

    def _compute_df_hash(self, df: pd.DataFrame) -> str:
        """
        Compute hash of DataFrame content.

        Uses pandas built-in hashing utility for efficiency.

        Args:
            df: Input DataFrame to hash

        Returns:
            Hash string (first 16 chars of SHA256)
        """
        vals = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.sha256(np.asarray(vals).tobytes()).hexdigest()[:16]

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
            self._evict_oldest(self.models_dir)
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

    @staticmethod
    def _get_labels_fallback_path(parquet_path: Path) -> Path:
        """Fallback path when parquet engines are unavailable."""
        return parquet_path.with_suffix(".pkl")

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

        fallback_path = self._get_labels_fallback_path(path)
        if fallback_path.exists():
            try:
                logger.info(f"Loading cached labels from {fallback_path.name} (pickle fallback)")
                return pd.read_pickle(fallback_path)
            except Exception as e:
                logger.warning(f"Failed to load fallback cached labels: {e}")
        return None

    def load_labels_memmap(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        columns: Optional[list[str]] = None,
        dtype: Any = np.float32,
    ) -> Optional[tuple[np.memmap, list[str]]]:
        """Load cached labels and expose selected columns as a read-only memmap array."""
        try:
            cached_df = self.load_labels(df, config)
            if cached_df is None:
                return None
            if columns is not None:
                cached_df = cached_df[list(columns)]

            path = self.get_labels_path(df, config)
            memmap_path = self.labels_dir / f"{path.stem}.mmap"
            mapped, used_columns = dataframe_to_memmap(
                cached_df,
                memmap_path,
                columns=cached_df.columns.tolist(),
                dtype=dtype,
            )
            logger.info(f"Loaded memory-mapped labels from {memmap_path.name}")
            return mapped, used_columns
        except Exception as e:
            logger.warning(f"Failed to load memory-mapped labels: {e}")
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
            self._evict_oldest(self.labels_dir)
        except Exception as e:
            logger.warning(f"Failed to save labels as parquet, trying pickle fallback: {e}")
            fallback_path = self._get_labels_fallback_path(path)
            try:
                labeled_df.to_pickle(fallback_path)
                logger.info(f"Saved labels to cache: {fallback_path.name} (pickle fallback)")
                self._evict_oldest(self.labels_dir)
            except Exception as fallback_error:
                logger.error(f"Failed to save labels to fallback cache: {fallback_error}")

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
