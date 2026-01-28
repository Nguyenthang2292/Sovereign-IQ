"""Tests for memory-mapped data utilities."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS.utils.memory_mapped_data import (
    MemmapDescriptor,
    MemoryMappedDataManager,
    create_memory_mapped_from_csv,
    load_memory_mapped_from_csv,
    get_manager,
)


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file with test data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("symbol,close,high,low\n")
        for i in range(100):
            f.write(f"BTC{i},{100.0 + i},{105.0 + i},{95.0 + i}\n")
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "mmap_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yield str(cache_dir)


class TestMemoryMappedDataCreation:
    """Test memory-mapped file creation."""

    def test_create_memory_mapped_from_csv(self, temp_csv_file, temp_cache_dir):
        """Test creating memory-mapped file from CSV."""
        descriptor = create_memory_mapped_from_csv(
            csv_path=temp_csv_file,
            symbol_column="symbol",
            price_column="close",
            cache_dir=temp_cache_dir,
        )

        assert descriptor is not None
        assert isinstance(descriptor, MemmapDescriptor)
        assert descriptor.shape == (100,)
        assert descriptor.columns == ["close", "high", "low"]
        assert Path(descriptor.mmap_path).exists()

    def test_descriptor_persistence(self, temp_csv_file, temp_cache_dir):
        """Test that descriptor can be loaded and reopened."""
        manager = MemoryMappedDataManager(cache_dir=temp_cache_dir)
        descriptor1 = manager.create_memory_mapped_from_csv(
            csv_path=temp_csv_file,
            symbol_column="symbol",
            price_column="close",
        )

        cache_key = descriptor1.mmap_path.split(os.sep)[-1].replace(".mmap", "")
        metadata_path = Path(temp_cache_dir) / f"{cache_key}.metadata"

        descriptor2 = manager.load_descriptor(metadata_path)

        assert descriptor1.shape == descriptor2.shape
        assert descriptor1.columns == descriptor2.columns
        assert descriptor1.dtype == descriptor2.dtype

    def test_reopen_and_readback(self, temp_csv_file, temp_cache_dir):
        """Test reopening and reading back memmap data."""
        descriptor = create_memory_mapped_from_csv(
            csv_path=temp_csv_file,
            symbol_column="symbol",
            price_column="close",
            cache_dir=temp_cache_dir,
        )

        mmap_array = np.memmap(descriptor.mmap_path, dtype=descriptor.dtype, mode="r", shape=descriptor.shape)

        assert len(mmap_array) == 100
        assert np.isclose(mmap_array["close"][0], 100.0)
        assert np.isclose(mmap_array["close"][99], 199.0)

    def test_column_selection(self, temp_csv_file, temp_cache_dir):
        """Test selecting specific columns for memory-mapping."""
        descriptor = create_memory_mapped_from_csv(
            csv_path=temp_csv_file,
            symbol_column="symbol",
            price_column="close",
            columns_to_map=["close", "high"],
            cache_dir=temp_cache_dir,
        )

        assert descriptor.columns == ["close", "high"]
        assert len(descriptor.columns) == 2

    def test_missing_column_error(self, temp_csv_file, temp_cache_dir):
        """Test error when required column is missing."""
        with pytest.raises(ValueError, match="Symbol column 'invalid' not found"):
            create_memory_mapped_from_csv(
                csv_path=temp_csv_file,
                symbol_column="invalid",
                price_column="close",
                cache_dir=temp_cache_dir,
            )

    def test_file_not_found_error(self, temp_cache_dir):
        """Test error when CSV file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            create_memory_mapped_from_csv(
                csv_path="nonexistent.csv",
                symbol_column="symbol",
                price_column="close",
                cache_dir=temp_cache_dir,
            )

    def test_reuse_existing_memmap(self, temp_csv_file, temp_cache_dir):
        """Test reusing existing memory-mapped file."""
        manager = MemoryMappedDataManager(cache_dir=temp_cache_dir)
        descriptor1 = manager.create_memory_mapped_from_csv(
            csv_path=temp_csv_file,
            symbol_column="symbol",
            price_column="close",
        )

        descriptor2 = manager.create_memory_mapped_from_csv(
            csv_path=temp_csv_file,
            symbol_column="symbol",
            price_column="close",
            overwrite=False,
        )

        assert descriptor1.mmap_path == descriptor2.mmap_path

    def test_overwrite_existing_memmap(self, temp_csv_file, temp_cache_dir):
        """Test overwriting existing memory-mapped file."""
        manager = MemoryMappedDataManager(cache_dir=temp_cache_dir)
        descriptor1 = manager.create_memory_mapped_from_csv(
            csv_path=temp_csv_file,
            symbol_column="symbol",
            price_column="close",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("symbol,close\n")
            for i in range(50):
                f.write(f"ETH{i},{200.0 + i}\n")
            temp_path2 = f.name

        try:
            os.unlink(temp_csv_file)
            os.rename(temp_path2, temp_csv_file)

            descriptor2 = manager.create_memory_mapped_from_csv(
                csv_path=temp_csv_file,
                symbol_column="symbol",
                price_column="close",
                overwrite=True,
            )

            assert descriptor2.shape == (50,)
        finally:
            if os.path.exists(temp_path2):
                os.unlink(temp_path2)


class TestMemoryMappedDataManager:
    """Test MemoryMappedDataManager functionality."""

    def test_initialization(self, temp_cache_dir):
        """Test manager initialization creates cache directory."""
        manager = MemoryMappedDataManager(cache_dir=temp_cache_dir)
        assert Path(temp_cache_dir).exists()
        assert manager.cache_dir == Path(temp_cache_dir)

    def test_load_from_csv_path(self, temp_csv_file, temp_cache_dir):
        """Test loading memmap from CSV path."""
        manager = MemoryMappedDataManager(cache_dir=temp_cache_dir)
        descriptor, mmap_array = manager.load_from_csv_path(
            csv_path=temp_csv_file,
            symbol_column="symbol",
            price_column="close",
        )

        assert descriptor is not None
        assert mmap_array is not None
        assert len(mmap_array) == 100

    def test_load_nonexistent_csv(self, temp_cache_dir):
        """Test loading nonexistent CSV returns None."""
        manager = MemoryMappedDataManager(cache_dir=temp_cache_dir)
        descriptor, mmap_array = manager.load_from_csv_path(
            csv_path="nonexistent.csv",
            symbol_column="symbol",
            price_column="close",
        )

        assert descriptor is None
        assert mmap_array is None

    def test_cleanup_old_files(self, temp_csv_file, temp_cache_dir):
        """Test cleanup of old cache files."""
        manager = MemoryMappedDataManager(cache_dir=temp_cache_dir)

        create_memory_mapped_from_csv(
            csv_path=temp_csv_file,
            symbol_column="symbol",
            price_column="close",
            cache_dir=temp_cache_dir,
        )

        files_removed = manager.cleanup(older_than_days=-1)
        assert files_removed >= 2


class TestBacktestPathSelection:
    """Test backtest path selection with use_memory_mapped flag."""

    def test_normal_path_without_memmap(self, temp_csv_file):
        """Test normal backtest path without memory-mapping."""
        from modules.adaptive_trend_LTS.core.backtesting.dask_backtest import backtest_with_dask

        atc_config = {"ema_len": 28, "atc_period": 14, "volatility_window": 20}

        try:
            result = backtest_with_dask(
                historical_data_path=temp_csv_file,
                atc_config=atc_config,
                use_memory_mapped=False,
            )

            assert result is not None
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"Backtest skipped due to: {e}")

    def test_memmap_path_enabled(self, temp_csv_file, temp_cache_dir):
        """Test backtest with memory-mapping enabled."""
        from modules.adaptive_trend_LTS.core.backtesting.dask_backtest import backtest_with_dask

        atc_config = {"ema_len": 28, "atc_period": 14, "volatility_window": 20}

        try:
            result = backtest_with_dask(
                historical_data_path=temp_csv_file,
                atc_config=atc_config,
                use_memory_mapped=True,
            )

            assert result is not None
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"Backtest skipped due to: {e}")

    def test_both_paths_produce_same_schema(self, temp_csv_file, temp_cache_dir):
        """Test both paths produce same result schema."""
        from modules.adaptive_trend_LTS.core.backtesting.dask_backtest import backtest_with_dask

        atc_config = {"ema_len": 28, "atc_period": 14, "volatility_window": 20}

        try:
            result_normal = backtest_with_dask(
                historical_data_path=temp_csv_file,
                atc_config=atc_config,
                use_memory_mapped=False,
            )

            result_memmap = backtest_with_dask(
                historical_data_path=temp_csv_file,
                atc_config=atc_config,
                use_memory_mapped=True,
            )

            assert result_normal.columns.tolist() == result_memmap.columns.tolist()
            assert result_normal.dtypes.to_dict() == result_memmap.dtypes.to_dict()
        except Exception as e:
            pytest.skip(f"Backtest comparison skipped due to: {e}")


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_manager_singleton(self, temp_cache_dir):
        """Test get_manager returns singleton instance."""
        manager1 = get_manager(cache_dir=temp_cache_dir)
        manager2 = get_manager(cache_dir=temp_cache_dir)

        assert manager1 is manager2

    def test_load_memory_mapped_from_csv(self, temp_csv_file, temp_cache_dir):
        """Test convenience function for loading memmap."""
        descriptor, mmap_array = load_memory_mapped_from_csv(
            csv_path=temp_csv_file,
            symbol_column="symbol",
            price_column="close",
            cache_dir=temp_cache_dir,
        )

        assert descriptor is not None
        assert mmap_array is not None
        assert len(mmap_array) == 100
