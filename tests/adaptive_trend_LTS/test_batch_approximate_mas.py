"""Comprehensive tests for BatchApproximateMAScanner."""

import pytest
import pandas as pd
import numpy as np
from typing import Dict

from modules.adaptive_trend_LTS.core.compute_moving_averages.batch_approximate_mas import (
    BatchApproximateMAScanner,
)


class TestBatchApproximateMAScannerInit:
    """Test initialization and configuration."""

    def test_default_initialization(self):
        """Test scanner initialization with default parameters."""
        scanner = BatchApproximateMAScanner()
        assert scanner.use_adaptive is False
        assert scanner.num_threads == 4
        assert scanner.volatility_window == 20
        assert scanner.base_tolerance == 0.05
        assert scanner.volatility_factor == 1.0
        assert scanner.get_symbol_count() == 0

    def test_adaptive_mode_initialization(self):
        """Test scanner initialization with adaptive mode enabled."""
        scanner = BatchApproximateMAScanner(
            use_adaptive=True,
            num_threads=8,
            volatility_window=10,
            base_tolerance=0.1,
            volatility_factor=2.0,
        )
        assert scanner.use_adaptive is True
        assert scanner.num_threads == 8
        assert scanner.volatility_window == 10
        assert scanner.base_tolerance == 0.1
        assert scanner.volatility_factor == 2.0


class TestBatchApproximateMAScannerAddRemove:
    """Test adding and removing symbols."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        return pd.Series(
            np.random.randn(100).cumsum() + 100,
            index=pd.date_range("2024-01-01", periods=100),
        )

    def test_add_single_symbol(self, sample_prices):
        """Test adding a single symbol."""
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", sample_prices)
        assert scanner.get_symbol_count() == 1
        assert "BTCUSDT" in scanner.get_symbols()

    def test_add_multiple_symbols(self, sample_prices):
        """Test adding multiple symbols."""
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", sample_prices)
        scanner.add_symbol("ETHUSDT", sample_prices * 0.5 + 50)
        scanner.add_symbol("SOLUSDT", sample_prices * 0.2 + 20)
        assert scanner.get_symbol_count() == 3
        assert set(scanner.get_symbols()) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}

    def test_replace_existing_symbol(self, sample_prices):
        """Test replacing an existing symbol."""
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", sample_prices)
        original_count = scanner.get_symbol_count()
        scanner.add_symbol("BTCUSDT", sample_prices * 2)
        assert scanner.get_symbol_count() == original_count
        assert "BTCUSDT" in scanner.get_symbols()

    def test_remove_symbol(self, sample_prices):
        """Test removing a symbol."""
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", sample_prices)
        scanner.add_symbol("ETHUSDT", sample_prices)
        removed = scanner.remove_symbol("BTCUSDT")
        assert removed is True
        assert scanner.get_symbol_count() == 1
        assert "BTCUSDT" not in scanner.get_symbols()

    def test_remove_nonexistent_symbol(self, sample_prices):
        """Test removing a symbol that doesn't exist."""
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", sample_prices)
        removed = scanner.remove_symbol("ETHUSDT")
        assert removed is False
        assert scanner.get_symbol_count() == 1

    def test_add_empty_prices(self):
        """Test adding a symbol with empty price data."""
        scanner = BatchApproximateMAScanner()
        empty_prices = pd.Series()
        scanner.add_symbol("BTCUSDT", empty_prices)
        assert scanner.get_symbol_count() == 0


class TestBatchApproximateMAScannerCalculateSingle:
    """Test calculating approximate MA for single symbols."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        return pd.Series(
            np.random.randn(100).cumsum() + 100,
            index=pd.date_range("2024-01-01", periods=100),
        )

    @pytest.fixture
    def scanner(self, sample_prices):
        """Create scanner with sample data."""
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", sample_prices)
        scanner.add_symbol("ETHUSDT", sample_prices * 0.5 + 50)
        return scanner

    @pytest.mark.parametrize(
        "ma_type, length", [("EMA", 20), ("HMA", 20), ("WMA", 20), ("DEMA", 20), ("LSMA", 20), ("KAMA", 20)]
    )
    def test_calculate_all_ma_types(self, scanner, ma_type, length):
        """Test calculating all 6 MA types."""
        results = scanner.calculate_all(ma_type, length)
        assert len(results) == 2
        assert "BTCUSDT" in results
        assert "ETHUSDT" in results
        assert len(results["BTCUSDT"]) == 100
        assert len(results["ETHUSDT"]) == 100

    @pytest.mark.parametrize("ma_type", ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"])
    def test_calculate_single_symbol(self, scanner, ma_type):
        """Test calculating MA for a single symbol."""
        result = scanner.calculate_symbol("BTCUSDT", ma_type, 20)
        assert result is not None
        assert len(result) == 100

    def test_calculate_nonexistent_symbol(self, scanner):
        """Test calculating MA for a symbol that doesn't exist."""
        result = scanner.calculate_symbol("SOLUSDT", "EMA", 20)
        assert result is None

    def test_calculate_invalid_ma_type(self, scanner):
        """Test calculating with an invalid MA type."""
        results = scanner.calculate_all("INVALID", 20)
        assert len(results) == 0


class TestBatchApproximateMAScannerAdaptiveMode:
    """Test adaptive mode functionality."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        return pd.Series(
            np.random.randn(100).cumsum() + 100,
            index=pd.date_range("2024-01-01", periods=100),
        )

    def test_adaptive_mode_enabled(self, sample_prices):
        """Test calculation with adaptive mode enabled."""
        scanner = BatchApproximateMAScanner(use_adaptive=True)
        scanner.add_symbol("BTCUSDT", sample_prices)
        results = scanner.calculate_all("EMA", 20)
        assert len(results) == 1
        assert "BTCUSDT" in results

    def test_adaptive_mode_disabled(self, sample_prices):
        """Test calculation with adaptive mode disabled."""
        scanner = BatchApproximateMAScanner(use_adaptive=False)
        scanner.add_symbol("BTCUSDT", sample_prices)
        results = scanner.calculate_all("EMA", 20)
        assert len(results) == 1
        assert "BTCUSDT" in results


class TestBatchApproximateMAScannerResults:
    """Test result retrieval and caching."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        return pd.Series(
            np.random.randn(100).cumsum() + 100,
            index=pd.date_range("2024-01-01", periods=100),
        )

    def test_get_all_results(self, sample_prices):
        """Test getting all cached results."""
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", sample_prices)
        scanner.add_symbol("ETHUSDT", sample_prices * 0.5 + 50)
        scanner.calculate_all("EMA", 20)
        scanner.calculate_all("HMA", 20)

        results = scanner.get_all_results()
        assert len(results) == 2
        assert "BTCUSDT" in results
        assert "ETHUSDT" in results
        assert ("EMA", 20) in results["BTCUSDT"]
        assert ("HMA", 20) in results["BTCUSDT"]

    def test_get_symbol_results(self, sample_prices):
        """Test getting results for a specific symbol."""
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", sample_prices)
        scanner.add_symbol("ETHUSDT", sample_prices * 0.5 + 50)
        scanner.calculate_all("EMA", 20)

        btc_results = scanner.get_symbol_results("BTCUSDT")
        assert btc_results is not None
        assert ("EMA", 20) in btc_results

    def test_get_symbol_result(self, sample_prices):
        """Test getting a specific result."""
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", sample_prices)
        scanner.calculate_all("EMA", 20)

        result = scanner.get_symbol_result("BTCUSDT", "EMA", 20)
        assert result is not None
        assert len(result) == 100

    def test_get_nonexistent_symbol_result(self, sample_prices):
        """Test getting result for a symbol that doesn't exist."""
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", sample_prices)
        scanner.calculate_all("EMA", 20)

        result = scanner.get_symbol_result("ETHUSDT", "EMA", 20)
        assert result is None

    def test_reset(self, sample_prices):
        """Test resetting all cached results."""
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", sample_prices)
        scanner.calculate_all("EMA", 20)
        scanner.calculate_all("HMA", 20)

        assert len(scanner.get_all_results()) > 0
        scanner.reset()
        assert len(scanner.get_all_results()) == 0


class TestBatchApproximateMAScannerMASets:
    """Test calculating sets of MAs (9 MAs per symbol)."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        return pd.Series(
            np.random.randn(100).cumsum() + 100,
            index=pd.date_range("2024-01-01", periods=100),
        )

    def test_calculate_set_of_mas(self, sample_prices):
        """Test calculating a set of 9 MAs."""
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", sample_prices)
        scanner.add_symbol("ETHUSDT", sample_prices * 0.5 + 50)

        results = scanner.calculate_set_of_mas("EMA", 20, robustness="Medium")
        assert results is not None
        assert len(results) == 2
        assert "BTCUSDT" in results
        assert "ETHUSDT" in results
        assert len(results["BTCUSDT"]) == 9
        assert len(results["ETHUSDT"]) == 9

    @pytest.mark.parametrize("robustness", ["Narrow", "Medium", "Wide"])
    def test_calculate_set_with_different_robustness(self, sample_prices, robustness):
        """Test calculating MA sets with different robustness levels."""
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", sample_prices)

        results = scanner.calculate_set_of_mas("EMA", 20, robustness=robustness)
        assert results is not None
        assert "BTCUSDT" in results
        assert len(results["BTCUSDT"]) == 9


class TestBatchApproximateMAScannerPerformance:
    """Test performance with large batches."""

    def test_large_batch_processing(self):
        """Test processing 100 symbols."""
        np.random.seed(42)
        scanner = BatchApproximateMAScanner(num_threads=4)

        for i in range(100):
            prices = pd.Series(
                np.random.randn(100).cumsum() + 100,
                index=pd.date_range("2024-01-01", periods=100),
            )
            scanner.add_symbol(f"SYMBOL{i}", prices)

        assert scanner.get_symbol_count() == 100

        results = scanner.calculate_all("EMA", 20, use_parallel=True)
        assert len(results) == 100

    def test_serial_vs_parallel(self):
        """Test serial vs parallel processing speed."""
        np.random.seed(42)

        symbols = {}
        for i in range(20):
            prices = pd.Series(
                np.random.randn(100).cumsum() + 100,
                index=pd.date_range("2024-01-01", periods=100),
            )
            symbols[f"SYMBOL{i}"] = prices

        scanner_parallel = BatchApproximateMAScanner(num_threads=4)
        for symbol, prices in symbols.items():
            scanner_parallel.add_symbol(symbol, prices)

        import time

        start = time.time()
        results_parallel = scanner_parallel.calculate_all("EMA", 20, use_parallel=True)
        parallel_time = time.time() - start

        scanner_serial = BatchApproximateMAScanner(num_threads=4)
        for symbol, prices in symbols.items():
            scanner_serial.add_symbol(symbol, prices)

        start = time.time()
        results_serial = scanner_serial.calculate_all("EMA", 20, use_parallel=False)
        serial_time = time.time() - start

        assert len(results_parallel) == len(results_serial)
        print(f"Parallel time: {parallel_time:.3f}s, Serial time: {serial_time:.3f}s")


class TestBatchApproximateMAScannerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_scanner(self):
        """Test operations on an empty scanner."""
        scanner = BatchApproximateMAScanner()
        results = scanner.calculate_all("EMA", 20)
        assert len(results) == 0

    def test_zero_length_prices(self):
        """Test with zero length prices."""
        prices = pd.Series([], dtype=float)
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", prices)
        assert scanner.get_symbol_count() == 0

    def test_single_price_point(self):
        """Test with a single price point."""
        prices = pd.Series([100.0])
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", prices)
        result = scanner.calculate_symbol("BTCUSDT", "EMA", 20)
        assert result is not None
        assert len(result) == 1

    def test_remove_symbol_with_results(self):
        """Test that removing a symbol also removes its results."""
        np.random.seed(42)
        prices = pd.Series(
            np.random.randn(100).cumsum() + 100,
            index=pd.date_range("2024-01-01", periods=100),
        )
        scanner = BatchApproximateMAScanner()
        scanner.add_symbol("BTCUSDT", prices)
        scanner.calculate_all("EMA", 20)

        assert "BTCUSDT" in scanner.get_all_results()
        scanner.remove_symbol("BTCUSDT")
        assert "BTCUSDT" not in scanner.get_all_results()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
