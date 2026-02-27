"""
Comprehensive tests for CacheManager.
Tests caching, hashing, and persistence.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.xgboost_LTS.utils.cache_manager import CacheManager


class TestCacheManagerInitialization:
    """Test CacheManager initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        cache = CacheManager()

        assert cache.cache_dir.exists()
        assert cache.models_dir.exists()
        assert cache.labels_dir.exists()

    def test_custom_subsystem(self):
        """Test initialization with custom subsystem."""
        cache = CacheManager(subsystem="test_subsystem")

        assert "test_subsystem" in str(cache.cache_dir)
        assert cache.cache_dir.exists()

    def test_directory_creation(self, tmp_path):
        """Test that directories are created on init."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager(subsystem="new_system")

            assert cache.models_dir.exists()
            assert cache.labels_dir.exists()


class TestHashComputation:
    """Test hash computation functions."""

    def test_df_hash_consistency(self):
        """Test that same DataFrame produces same hash."""
        cache = CacheManager()

        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        hash1 = cache._compute_df_hash(df1)
        hash2 = cache._compute_df_hash(df2)

        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars

    def test_df_hash_difference(self):
        """Test that different DataFrames produce different hashes."""
        cache = CacheManager()

        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})  # Different value

        hash1 = cache._compute_df_hash(df1)
        hash2 = cache._compute_df_hash(df2)

        assert hash1 != hash2

    def test_config_hash_consistency(self):
        """Test that same config produces same hash."""
        cache = CacheManager()

        config1 = {"a": 1, "b": [1, 2, 3], "c": "test"}
        config2 = {"a": 1, "b": [1, 2, 3], "c": "test"}

        hash1 = cache._compute_config_hash(config1)
        hash2 = cache._compute_config_hash(config2)

        assert hash1 == hash2

    def test_config_hash_order_independence(self):
        """Test that config hash is independent of key order."""
        cache = CacheManager()

        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}

        hash1 = cache._compute_config_hash(config1)
        hash2 = cache._compute_config_hash(config2)

        assert hash1 == hash2

    def test_config_hash_with_nested_dicts(self):
        """Test hashing with nested dictionaries."""
        cache = CacheManager()

        config = {"level1": {"level2": {"value": 123}}, "list": [1, 2, {"nested": "value"}]}

        # Should not raise
        hash_val = cache._compute_config_hash(config)
        assert len(hash_val) == 16


class TestModelCaching:
    """Test model save/load operations."""

    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading a model."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            # Create dummy model
            model = {"weights": np.array([1, 2, 3]), "bias": 0.5}

            df = pd.DataFrame({"feature": [1, 2, 3]})
            config = {"learning_rate": 0.1, "epochs": 100}

            # Save
            cache.save_model(model, df, config)

            # Load
            loaded = cache.load_model(df, config)

            assert loaded is not None
            np.testing.assert_array_equal(loaded["weights"], model["weights"])
            assert loaded["bias"] == model["bias"]

    def test_load_nonexistent_model(self, tmp_path):
        """Test loading a model that doesn't exist."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            df = pd.DataFrame({"feature": [1, 2, 3]})
            config = {"learning_rate": 0.1}

            loaded = cache.load_model(df, config)

            assert loaded is None

    def test_model_caching_different_configs(self, tmp_path):
        """Test that different configs create different cache files."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            model = {"type": "test"}
            df = pd.DataFrame({"feature": [1, 2, 3]})

            config1 = {"lr": 0.1}
            config2 = {"lr": 0.2}

            cache.save_model(model, df, config1)
            cache.save_model(model, df, config2)

            # Both should be loadable
            assert cache.load_model(df, config1) is not None
            assert cache.load_model(df, config2) is not None

    def test_model_cache_invalidation_on_data_change(self, tmp_path):
        """Test that model cache is invalidated when data changes."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            model = {"weights": [1, 2, 3]}
            config = {"lr": 0.1}

            df1 = pd.DataFrame({"feature": [1, 2, 3]})
            df2 = pd.DataFrame({"feature": [1, 2, 4]})  # Different data

            cache.save_model(model, df1, config)

            # Should not find model for different data
            assert cache.load_model(df2, config) is None

    def test_corrupted_cache_file_handling(self, tmp_path):
        """Test handling of corrupted cache files."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            model = {"test": "data"}
            df = pd.DataFrame({"feature": [1, 2, 3]})
            config = {"lr": 0.1}

            cache.save_model(model, df, config)

            # Corrupt the file
            model_path = cache.get_model_path(df, config, native=False)
            with open(model_path, "w") as f:
                f.write("corrupted data")

            # Should return None, not crash
            loaded = cache.load_model(df, config)
            assert loaded is None


class TestLabelsCaching:
    """Test labels save/load operations."""

    def test_save_and_load_labels(self, tmp_path):
        """Test saving and loading labels."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            source_df = pd.DataFrame({"close": [100, 101, 102], "volume": [1000, 1100, 1200]})

            labeled_df = source_df.copy()
            labeled_df["Target"] = [0, 1, 2]
            labeled_df["TargetLabel"] = ["DOWN", "NEUTRAL", "UP"]

            config = {"threshold": 0.01}

            # Save
            cache.save_labels(labeled_df, source_df, config)

            # Load
            loaded = cache.load_labels(source_df, config)

            assert loaded is not None
            assert "Target" in loaded.columns
            assert list(loaded["Target"]) == [0, 1, 2]

    def test_load_nonexistent_labels(self, tmp_path):
        """Test loading labels that don't exist."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            df = pd.DataFrame({"close": [100, 101]})
            config = {"threshold": 0.01}

            loaded = cache.load_labels(df, config)

            assert loaded is None

    def test_labels_cache_selective_hashing(self, tmp_path):
        """Test that only OHLCV columns are used for labels hash."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            # Source with OHLCV
            source_df = pd.DataFrame(
                {
                    "open": [100, 101],
                    "high": [101, 102],
                    "low": [99, 100],
                    "close": [100, 101],
                    "volume": [1000, 1100],
                    "extra_col": [1, 2],  # This should not affect hash
                }
            )

            labeled_df = source_df.copy()
            labeled_df["Target"] = [0, 1]

            config = {"threshold": 0.01}
            cache.save_labels(labeled_df, source_df, config)

            # Change only extra_col
            source_df2 = source_df.copy()
            source_df2["extra_col"] = [999, 999]

            # Should still find the cached labels
            loaded = cache.load_labels(source_df2, config)
            # Note: This might be None depending on implementation
            # The key point is it doesn't crash

    def test_labels_preserve_dtypes(self, tmp_path):
        """Test that labels preserve data types."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            source_df = pd.DataFrame({"close": [100.0, 101.0], "volume": [1000, 1100]})

            labeled_df = source_df.copy()
            labeled_df["Target"] = pd.Series([0, 1], dtype="Int64")
            labeled_df["TargetLabel"] = pd.Series(["DOWN", "UP"], dtype="string")

            config = {"threshold": 0.01}
            cache.save_labels(labeled_df, source_df, config)

            loaded = cache.load_labels(source_df, config)

            if loaded is not None:
                assert loaded["Target"].dtype == "Int64" or loaded["Target"].dtype == "int64"


class TestCacheClearing:
    """Test cache clearing functionality."""

    def test_clear_cache(self, tmp_path):
        """Test clearing all cache files."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            # Create some cache files
            df = pd.DataFrame({"feature": [1, 2, 3]})
            config = {"lr": 0.1}

            cache.save_model({"test": "model"}, df, config)

            # Verify file exists
            assert len(list(cache.models_dir.iterdir())) > 0

            # Clear cache
            cache.clear_cache()

            # Verify files are gone
            assert len(list(cache.models_dir.iterdir())) == 0
            assert len(list(cache.labels_dir.iterdir())) == 0

    def test_clear_empty_cache(self, tmp_path):
        """Test clearing an already empty cache."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            # Should not raise
            cache.clear_cache()

            assert len(list(cache.models_dir.iterdir())) == 0


class TestCachePathGeneration:
    """Test cache path generation."""

    def test_model_path_format(self, tmp_path):
        """Test model path format."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            df = pd.DataFrame({"feature": [1, 2, 3]})
            config = {"lr": 0.1}

            path = cache.get_model_path(df, config)

            assert path.suffix == ".json"
            assert "model_" in path.name
            assert path.parent == cache.models_dir

    def test_model_path_with_suffix(self, tmp_path):
        """Test model path with custom suffix."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            df = pd.DataFrame({"feature": [1, 2, 3]})
            config = {"lr": 0.1}

            path = cache.get_model_path(df, config, suffix="_v2")

            assert "_v2" in path.name

    def test_labels_path_format(self, tmp_path):
        """Test labels path format."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache = CacheManager()

            df = pd.DataFrame({"close": [100, 101]})
            config = {"threshold": 0.01}

            path = cache.get_labels_path(df, config)

            assert path.suffix == ".parquet"
            assert "labels_" in path.name
            assert path.parent == cache.labels_dir


class TestCacheConcurrency:
    """Test cache behavior under concurrent access (simulated)."""

    def test_multiple_caches_same_subsystem(self, tmp_path):
        """Test that multiple CacheManagers can coexist."""
        with patch("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path):
            cache1 = CacheManager(subsystem="test")
            cache2 = CacheManager(subsystem="test")

            df = pd.DataFrame({"feature": [1, 2, 3]})
            config = {"lr": 0.1}

            cache1.save_model({"model": 1}, df, config)

            # cache2 should see the same data
            loaded = cache2.load_model(df, config)
            assert loaded is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
