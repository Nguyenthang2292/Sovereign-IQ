"""
Tests for Phase 3: Caching & Persistence optimizations in XGBoost LTS.
"""

import pytest
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Ensure modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from modules.xgboost_LTS.utils.cache_manager import CacheManager
from modules.xgboost_LTS.core.labeling import apply_directional_labels
from modules.xgboost_LTS.core.model import train_and_predict
from config import ARTIFACTS_DIR, MODEL_FEATURES


class TestCacheManager:
    """Test CacheManager functionality."""

    @pytest.fixture
    def cache_manager(self, tmp_path):
        """Create CacheManager with temporary directory."""
        # Patch ARTIFACTS_DIR to use tmp_path
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", str(tmp_path)):
            cm = CacheManager(subsystem="test_xgboost")
            yield cm

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({"A": range(10), "B": range(10, 20)})

    def test_compute_hashes(self, cache_manager, sample_df):
        """Verify hash computation is deterministic."""
        config = {"param1": 10, "param2": "test"}

        hash1 = cache_manager._compute_df_hash(sample_df)
        hash2 = cache_manager._compute_df_hash(sample_df.copy())

        assert hash1 == hash2
        assert len(hash1) == 16

        config_hash1 = cache_manager._compute_config_hash(config)
        config_hash2 = cache_manager._compute_config_hash(config.copy())

        assert config_hash1 == config_hash2

    def test_model_save_load(self, cache_manager, sample_df):
        """Verify saving and loading models."""
        config = {"model_type": "test"}
        model = {"state": 123}

        # Save
        cache_manager.save_model(model, sample_df, config)

        # Load
        loaded_model = cache_manager.load_model(sample_df, config)

        assert loaded_model is not None
        assert loaded_model == model

    def test_labels_save_load(self, cache_manager, sample_df):
        """Verify saving and loading labeled DataFrames."""
        config = {"horizon": 24}
        labeled_df = sample_df.copy()
        labeled_df["Label"] = "UP"

        # Save
        cache_manager.save_labels(labeled_df, sample_df, config)

        # Load
        loaded_df = cache_manager.load_labels(sample_df, config)

        assert loaded_df is not None
        pd.testing.assert_frame_equal(labeled_df, loaded_df)


class TestCachingIntegration:
    """Test integration of caching in core functions."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Temp dir for artifacts."""
        path = tmp_path / "artifacts"
        path.mkdir()
        return path

    @pytest.fixture
    def data(self):
        """Sample data for labeling/training."""
        rows = 100
        dates = pd.date_range("2023-01-01", periods=rows, freq="h")
        df = pd.DataFrame(
            {
                "close": np.random.randn(rows) + 100,
                "open": np.random.randn(rows) + 100,
                "high": np.random.randn(rows) + 105,
                "low": np.random.randn(rows) + 95,
                "volume": np.random.randint(100, 1000, rows),
                "ATR_14": np.random.rand(rows),
                "Target": np.random.randint(0, 3, rows),
            },
            index=dates,
        )

        # Ensure MODEL_FEATURES exist
        for col in MODEL_FEATURES:
            if col not in df.columns:
                df[col] = np.random.randn(rows)

        return df

    def test_labeling_caching(self, data, cache_dir):
        """Verify apply_directional_labels uses cache."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", str(cache_dir)):
            # First run: Should save to cache
            with patch("modules.xgboost_LTS.utils.cache_manager.CacheManager.save_labels") as mock_save:
                df1 = apply_directional_labels(data.copy(), use_cache=True)
                assert mock_save.called

            # Second run: Should load from cache
            # We need to actually save it first to test loading,
            # so let's run without mock save first
            apply_directional_labels(data.copy(), use_cache=True)

            with patch("modules.xgboost_LTS.utils.cache_manager.CacheManager.load_labels") as mock_load:
                # Mock return value to ensure it's used
                mock_load.return_value = df1
                df2 = apply_directional_labels(data.copy(), use_cache=True)
                assert mock_load.called
                pd.testing.assert_frame_equal(df1, df2)

    def test_training_caching(self, data, cache_dir):
        """Verify train_and_predict uses cache."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", str(cache_dir)):
            # First run
            model1 = train_and_predict(data.copy(), use_cache=True)

            # Second run - should load from cache
            # We can verify this by checking logs or patching load_model
            with patch("modules.xgboost_LTS.utils.cache_manager.CacheManager.load_model") as mock_load:
                mock_load.return_value = model1
                model2 = train_and_predict(data.copy(), use_cache=True)
                assert mock_load.called
                assert model2 == model1
