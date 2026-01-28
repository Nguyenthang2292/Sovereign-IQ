"""Tests for data compression utilities and CacheManager compression."""

import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS.utils.cache_manager import CacheManager
from modules.adaptive_trend_LTS.utils.data_compression import (
    BLOSC_AVAILABLE,
    compress_pickle,
    compress_to_file,
    decompress_from_file,
    decompress_pickle,
    get_compression_ratio,
    get_default_compression_level,
    get_supported_algorithms,
    is_compression_available,
    validate_compression_params,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yield str(cache_dir)


class TestDataCompressionUtilities:
    """Test compression utility functions."""

    def test_compression_availability(self):
        """Test that compression availability is correctly detected."""
        is_available = is_compression_available()
        assert isinstance(is_available, bool)

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_compress_decompress_roundtrip(self):
        """Test compress/decompress roundtrip preserves data."""
        original_data = {
            "key1": np.array([1, 2, 3, 4, 5]),
            "key2": pd.Series([10, 20, 30, 40, 50]),
            "key3": "test string",
            "key4": 123.456,
        }

        compressed = compress_pickle(original_data)
        decompressed = decompress_pickle(compressed)

        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

        for key, value in original_data.items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(decompressed[key], value)
            elif isinstance(value, pd.Series):
                pd.testing.assert_series_equal(decompressed[key], value)
            else:
                assert decompressed[key] == value

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_different_compression_levels(self):
        """Test different compression levels with highly compressible data."""
        # Use highly compressible data (repeated pattern) to ensure level differences
        data = {"test": "A" * 10000}  # 10KB of identical characters

        compressed_1 = compress_pickle(data, compression_level=1)
        compressed_5 = compress_pickle(data, compression_level=5)
        compressed_9 = compress_pickle(data, compression_level=9)

        # All should be significantly smaller than original
        original_size = len(pickle.dumps(data))
        assert len(compressed_1) < original_size * 0.5  # At least 50% compression
        assert len(compressed_5) < original_size * 0.5
        assert len(compressed_9) < original_size * 0.5

        # Higher levels should achieve equal or better compression
        assert len(compressed_1) >= len(compressed_5)
        assert len(compressed_5) >= len(compressed_9)

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        data = {"test": "x" * 1000}

        compressed = compress_pickle(data)
        ratio = get_compression_ratio(len(pickle.dumps(data)), len(compressed))

        assert ratio > 1.0

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_compress_to_file(self):
        """Test compressing to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_cache")

            data = {"test": list(range(1000))}
            file_size = compress_to_file(data, filepath)

            assert os.path.exists(filepath + ".blosc")
            assert file_size > 0

            loaded_data = decompress_from_file(filepath)
            assert loaded_data == data

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_decompress_from_file_not_found(self):
        """Test error when file not found."""
        with pytest.raises(FileNotFoundError):
            decompress_from_file("/nonexistent/file.blosc")

    def test_get_default_compression_level(self):
        """Test get default compression level."""
        level = get_default_compression_level()
        assert 0 <= level <= 9

    def test_get_supported_algorithms(self):
        """Test get supported algorithms."""
        algorithms = get_supported_algorithms()

        if BLOSC_AVAILABLE:
            assert len(algorithms) > 0
            assert isinstance(algorithms[0], str)
        else:
            assert algorithms == []

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_validate_compression_params_valid(self):
        """Test validation with valid parameters."""
        is_valid, error = validate_compression_params(5, "blosclz")
        assert is_valid
        assert error is None

    def test_validate_compression_params_invalid_level(self):
        """Test validation with invalid compression level."""
        is_valid, error = validate_compression_params(15, "blosclz")
        assert not is_valid
        assert error is not None

    def test_validate_compression_params_invalid_algorithm(self):
        """Test validation with invalid algorithm."""
        is_valid, error = validate_compression_params(5, "invalid")
        assert not is_valid
        assert error is not None

    def test_validate_compression_params_no_blosc(self):
        """Test validation when blosc not available."""
        is_valid, error = validate_compression_params(5, "blosclz")

        if not BLOSC_AVAILABLE:
            assert not is_valid
            assert error is not None
        else:
            assert is_valid

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_compress_without_blosc(self):
        """Test error when blosc not available."""
        original_BLOSC_AVAILABLE = BLOSC_AVAILABLE

        if original_BLOSC_AVAILABLE:
            pytest.skip("blosc is installed, cannot test error case")

        with pytest.raises(RuntimeError, match="blosc is not available"):
            compress_pickle({"test": "data"})


class TestCacheManagerCompression:
    """Test CacheManager with compression."""

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_save_cache_compressed(self, temp_dir):
        """Test saving cache with compression enabled."""
        manager = CacheManager(cache_dir=temp_dir, use_compression=True)

        test_data = {"key1": np.array([1, 2, 3]), "key2": "test"}

        for key, value in test_data.items():
            manager.put("test", 28, np.array([1, 2, 3]), value, {"key": key})

        manager.save_to_disk()

        compressed_file = Path(temp_dir) / "cache_v1.pkl.blosc"
        assert compressed_file.exists()

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_save_cache_uncompressed(self, temp_dir):
        """Test saving cache without compression."""
        manager = CacheManager(cache_dir=temp_dir, use_compression=False)

        test_data = {"key1": np.array([1, 2, 3]), "key2": "test"}

        for key, value in test_data.items():
            manager.put("test", 28, np.array([1, 2, 3]), value, {"key": key})

        manager.save_to_disk()

        compressed_file = Path(temp_dir) / "cache_v1.pkl.blosc"
        uncompressed_file = Path(temp_dir) / "cache_v1.pkl"

        assert not compressed_file.exists()
        assert uncompressed_file.exists()

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_load_cache_compressed(self, temp_dir):
        """Test loading compressed cache."""
        manager1 = CacheManager(cache_dir=temp_dir, use_compression=True)

        test_data = {"key": np.array([1, 2, 3, 4, 5])}
        price_data = np.array([1, 2, 3])
        manager1.put("test", 28, price_data, test_data, {"test": "key"})

        # Access the cached item to increment hits (required for persistence)
        _ = manager1.get("test", 28, price_data, {"test": "key"})

        manager1.save_to_disk()

        manager2 = CacheManager(cache_dir=temp_dir, use_compression=True)
        manager2.load_from_disk()

        assert len(manager2._l2_cache) > 0

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_load_cache_uncompressed_fallback(self, temp_dir):
        """Test loading uncompressed cache when compression enabled."""
        manager1 = CacheManager(cache_dir=temp_dir, use_compression=False)

        test_data = {"key": "test_value"}
        price_data = np.array([1, 2, 3])
        manager1.put("test", 28, price_data, test_data, {"test": "key"})

        # Access the cached item to increment hits (required for persistence)
        _ = manager1.get("test", 28, price_data, {"test": "key"})

        manager1.save_to_disk()

        manager2 = CacheManager(cache_dir=temp_dir, use_compression=True)
        manager2.load_from_disk()

        assert len(manager2._l2_cache) > 0

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_compressed_cache_size_reduction(self, temp_dir):
        """Test that compression reduces cache file size."""
        uncompressed_manager = CacheManager(cache_dir=temp_dir, use_compression=False)
        compressed_manager = CacheManager(cache_dir=temp_dir, use_compression=True)

        test_data = {"key": "x" * 10000}
        price_data = np.array([1, 2, 3])

        uncompressed_manager.put("test", 28, price_data, test_data, {"test": "key"})
        # Access the cached item to increment hits (required for persistence)
        _ = uncompressed_manager.get("test", 28, price_data, {"test": "key"})
        uncompressed_manager.save_to_disk()

        compressed_manager.put("test", 28, price_data, test_data, {"test": "key"})
        # Access the cached item to increment hits (required for persistence)
        _ = compressed_manager.get("test", 28, price_data, {"test": "key"})
        compressed_manager.save_to_disk()

        uncompressed_file = Path(temp_dir) / "cache_v1.pkl"
        compressed_file = Path(temp_dir) / "cache_v1.pkl.blosc"

        if uncompressed_file.exists() and compressed_file.exists():
            uncompressed_size = os.path.getsize(uncompressed_file)
            compressed_size = os.path.getsize(compressed_file)

            assert compressed_size < uncompressed_size

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_backward_compatibility_with_old_cache(self, temp_dir):
        """Test backward compatibility with existing uncompressed cache."""
        from modules.adaptive_trend_LTS.utils.cache_manager import CacheEntry

        uncompressed_file = Path(temp_dir) / "cache_v1.pkl"

        # Create a proper CacheEntry object for backward compatibility
        key = "ma=test|len=28|d=abcdef1234567890"
        entry = CacheEntry(
            key=key,
            value={"old_key": "old_value"},
            timestamp=time.time(),
            hits=2,
            size_bytes=100,
            ma_type="test",
            length=28,
        )

        with open(uncompressed_file, "wb") as f:
            pickle.dump({key: entry}, f)

        manager = CacheManager(cache_dir=temp_dir, use_compression=True)
        manager.load_from_disk()

        assert len(manager._l2_cache) > 0

    def test_compression_disabled_when_blosc_unavailable(self, temp_dir):
        """Test compression auto-disabled when blosc unavailable."""
        manager = CacheManager(
            cache_dir=temp_dir,
            use_compression=True,  # Always request compression
        )

        # CacheManager should auto-disable compression if blosc not available
        if not BLOSC_AVAILABLE:
            assert not manager.use_compression
        else:
            assert manager.use_compression

        test_data = {"key": "value"}
        price_data = np.array([1, 2, 3])
        manager.put("test", 28, price_data, test_data, {"test": "key"})
        # Access to increment hits for persistence
        _ = manager.get("test", 28, price_data, {"test": "key"})
        manager.save_to_disk()

        compressed_file = Path(temp_dir) / "cache_v1.pkl.blosc"
        uncompressed_file = Path(temp_dir) / "cache_v1.pkl"

        if not BLOSC_AVAILABLE:
            assert not compressed_file.exists()
            assert uncompressed_file.exists()


class TestCacheManagerCompressionAlgorithms:
    """Test different compression algorithms."""

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_different_algorithms(self, temp_dir):
        """Test using different compression algorithms."""
        manager = CacheManager(
            cache_dir=temp_dir,
            use_compression=True,
            compression_algorithm="zstd",
        )

        test_data = {"key": "test_value"}
        manager.put("test", 28, np.array([1, 2, 3]), test_data, {"test": "key"})
        manager.save_to_disk()

        compressed_file = Path(temp_dir) / "cache_v1.pkl.blosc"
        assert compressed_file.exists()

    @pytest.mark.skipif(not BLOSC_AVAILABLE, reason="blosc not installed")
    def test_different_compression_levels(self, temp_dir):
        """Test using different compression levels."""
        manager = CacheManager(
            cache_dir=temp_dir,
            use_compression=True,
            compression_level=9,
        )

        test_data = {"key": "x" * 1000}
        manager.put("test", 28, np.array([1, 2, 3]), test_data, {"test": "key"})
        manager.save_to_disk()

        compressed_file = Path(temp_dir) / "cache_v1.pkl.blosc"
        assert compressed_file.exists()
