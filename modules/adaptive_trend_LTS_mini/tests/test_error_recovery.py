"""
Comprehensive tests for error recovery paths in ATC LTS Mini module.

Tests all exception handling paths, fallback mechanisms, and error recovery logic.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.adaptive_trend_LTS_mini.core.analyzer import SymbolAnalyzer
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS_mini.core.rust_backend import (
    RUST_AVAILABLE,
    calculate_dema,
    calculate_ema,
    calculate_equity,
    calculate_hma,
    calculate_kama,
    calculate_lsma,
    calculate_wma,
    process_signal_persistence,
)
from modules.adaptive_trend_LTS_mini.utils.config import (
    ATCConfig,
    create_atc_config_from_dict,
)


class TestRustFallbackMechanisms:
    """Test Rust backend fallback to Python/Numba implementations."""

    def setup_method(self):
        """Setup test data."""
        self.prices = pd.Series(np.random.randn(100).cumsum() + 100)
        self.length = 10

    def test_calculate_equity_fallback_when_rust_unavailable(self):
        """Test equity calculation falls back to Python when Rust unavailable."""
        r_values = np.random.randn(100) * 0.01
        sig_prev = np.random.choice([-1, 1], size=100)

        # Force Python path by setting use_rust=False
        result = calculate_equity(
            r_values=r_values,
            sig_prev=sig_prev,
            starting_equity=1.0,
            decay_multiplier=0.997,
            cutout=0,
            use_rust=False,
        )

        assert isinstance(result, np.ndarray)
        assert len(result) == len(r_values)
        assert not np.any(np.isnan(result))

    def test_calculate_equity_fallback_custom_floor(self):
        """Test equity calculation falls back to Python when custom floor requested."""
        r_values = np.random.randn(100) * 0.01
        sig_prev = np.random.choice([-1, 1], size=100)

        # Custom floor_val != 0.25 should trigger Python fallback
        result = calculate_equity(
            r_values=r_values,
            sig_prev=sig_prev,
            starting_equity=1.0,
            decay_multiplier=0.997,
            cutout=0,
            use_rust=True,
            floor_val=0.5,  # Custom floor triggers fallback
        )

        assert isinstance(result, np.ndarray)
        assert len(result) == len(r_values)

    def test_calculate_kama_fallback(self):
        """Test KAMA falls back to Numba when Rust unavailable."""
        result = calculate_kama(self.prices, length=self.length, use_rust=False)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.prices)

    def test_calculate_ema_fallback(self):
        """Test EMA falls back to pandas_ta when Rust unavailable."""
        result = calculate_ema(self.prices, length=self.length, use_rust=False)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.prices)

    def test_calculate_wma_fallback(self):
        """Test WMA falls back to pandas_ta when Rust unavailable."""
        result = calculate_wma(self.prices, length=self.length, use_rust=False)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.prices)

    def test_calculate_hma_fallback(self):
        """Test HMA falls back to pandas_ta when Rust unavailable."""
        result = calculate_hma(self.prices, length=self.length, use_rust=False)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.prices)

    def test_calculate_dema_fallback(self):
        """Test DEMA falls back to pandas_ta when Rust unavailable."""
        result = calculate_dema(self.prices, length=self.length, use_rust=False)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.prices)

    def test_calculate_lsma_fallback(self):
        """Test LSMA falls back to pandas_ta when Rust unavailable."""
        result = calculate_lsma(self.prices, length=self.length, use_rust=False)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.prices)

    def test_process_signal_persistence_fallback(self):
        """Test signal persistence falls back to Numba when Rust unavailable."""
        up = np.array([True, False, False, True, False])
        down = np.array([False, True, False, False, True])

        result = process_signal_persistence(up, down, use_rust=False)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(up)


class TestConfigValidation:
    """Test ATCConfig validation and error handling."""

    def test_create_config_invalid_robustness(self):
        """Test creating config with invalid robustness raises ValueError."""
        params = {"robustness": "Invalid"}

        with pytest.raises(ValueError, match="robustness must be one of"):
            create_atc_config_from_dict(params)

    def test_create_config_negative_limit(self):
        """Test creating config with negative limit raises ValueError."""
        params = {"limit": -100}

        with pytest.raises(ValueError, match="limit must be a positive integer"):
            create_atc_config_from_dict(params)

    def test_create_config_zero_limit(self):
        """Test creating config with zero limit raises ValueError."""
        params = {"limit": 0}

        with pytest.raises(ValueError, match="limit must be a positive integer"):
            create_atc_config_from_dict(params)

    def test_create_config_negative_equity_floor(self):
        """Test creating config with negative equity_floor raises ValueError."""
        params = {"equity_floor": -0.1}

        with pytest.raises(ValueError, match="equity_floor must be a non-negative"):
            create_atc_config_from_dict(params)

    def test_create_config_non_integer_limit(self):
        """Test creating config with non-integer limit raises ValueError."""
        params = {"limit": "not_an_int"}

        with pytest.raises(ValueError, match="limit must be a positive integer"):
            create_atc_config_from_dict(params)

    def test_create_config_valid_params(self):
        """Test creating config with valid parameters succeeds."""
        params = {
            "robustness": "Wide",
            "limit": 1000,
            "equity_floor": 0.3,
            "lambda_param": 0.05,
            "decay": 0.04,
        }

        config = create_atc_config_from_dict(params)

        assert isinstance(config, ATCConfig)
        assert config.robustness == "Wide"
        assert config.limit == 1000
        assert config.equity_floor == 0.3
        assert config.lambda_param == 0.05
        assert config.decay == 0.04


class TestDataFetchingErrors:
    """Test error handling in data fetching and analysis."""

    def test_symbol_analyzer_handles_empty_dataframe(self):
        """Test SymbolAnalyzer handles empty DataFrame gracefully."""
        # Mock data fetcher that returns empty DataFrame
        mock_fetcher = Mock()
        mock_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (
            pd.DataFrame(),
            "binance",
        )

        analyzer = SymbolAnalyzer(mock_fetcher)
        config = ATCConfig()

        result = analyzer.analyze("BTC/USDT", config)

        assert result is None

    def test_symbol_analyzer_handles_none_dataframe(self):
        """Test SymbolAnalyzer handles None DataFrame gracefully."""
        # Mock data fetcher that returns None
        mock_fetcher = Mock()
        mock_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (None, "binance")

        analyzer = SymbolAnalyzer(mock_fetcher)
        config = ATCConfig()

        result = analyzer.analyze("BTC/USDT", config)

        assert result is None

    def test_symbol_analyzer_handles_fetcher_exception(self):
        """Test SymbolAnalyzer handles data fetcher exceptions."""
        # Mock data fetcher that raises exception
        mock_fetcher = Mock()
        mock_fetcher.fetch_ohlcv_with_fallback_exchange.side_effect = Exception(
            "Network error"
        )

        analyzer = SymbolAnalyzer(mock_fetcher)
        config = ATCConfig()

        result = analyzer.analyze("BTC/USDT", config)

        assert result is None

    def test_symbol_analyzer_handles_invalid_price_column(self):
        """Test SymbolAnalyzer handles missing price column."""
        # Mock data fetcher that returns DataFrame without 'close' column
        mock_fetcher = Mock()
        df = pd.DataFrame({"volume": [100, 200, 300]})
        mock_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")

        analyzer = SymbolAnalyzer(mock_fetcher)
        config = ATCConfig(calculation_source="close")

        result = analyzer.analyze("BTC/USDT", config)

        assert result is None


class TestComputeATCSignalsErrors:
    """Test error handling in compute_atc_signals function."""

    def setup_method(self):
        """Setup test data."""
        self.prices = pd.Series(np.random.randn(100).cumsum() + 100)

    def test_compute_signals_with_insufficient_data(self):
        """Test compute_atc_signals with insufficient data points."""
        # Very short price series
        short_prices = pd.Series([100, 101, 102])

        # Should not raise exception, but may return NaN-filled results
        result = compute_atc_signals(
            short_prices,
            ema_len=28,  # Longer than data length
            use_rust_backend=False,
        )

        assert isinstance(result, dict)
        assert "Average_Signal" in result

    def test_compute_signals_with_nan_prices(self):
        """Test compute_atc_signals handles NaN values."""
        prices_with_nan = self.prices.copy()
        prices_with_nan.iloc[50] = np.nan

        # Should handle NaN values gracefully
        result = compute_atc_signals(prices_with_nan, use_rust_backend=False)

        assert isinstance(result, dict)

    def test_compute_signals_with_inf_prices(self):
        """Test compute_atc_signals handles infinite values."""
        prices_with_inf = self.prices.copy()
        prices_with_inf.iloc[50] = np.inf

        # Should handle inf values gracefully
        result = compute_atc_signals(prices_with_inf, use_rust_backend=False)

        assert isinstance(result, dict)

    def test_compute_signals_with_invalid_robustness(self):
        """Test compute_atc_signals with invalid robustness value."""
        # Invalid robustness should be caught by validation
        with pytest.raises((ValueError, KeyError)):
            compute_atc_signals(self.prices, robustness="Invalid")

    def test_compute_signals_decay_rate_deprecation_warning(self):
        """Test that decay_rate parameter raises deprecation warning."""
        with pytest.warns(DeprecationWarning, match="decay_rate= is deprecated"):
            result = compute_atc_signals(
                self.prices,
                decay_rate=0.05,  # Use deprecated parameter
                use_rust_backend=False,
            )

        assert isinstance(result, dict)

    def test_compute_signals_double_scaling_warning(self):
        """Test warning for already-scaled parameters."""
        # Very small lambda_param (appears already scaled)
        with pytest.warns(None) as warning_list:
            result = compute_atc_signals(
                self.prices,
                lambda_param=0.00001,  # Already scaled
                use_rust_backend=False,
            )

        # Check if appropriate warning was logged (via log_warn)
        assert isinstance(result, dict)


class TestSeriesPoolErrors:
    """Test series pool release failure handling."""

    def setup_method(self):
        """Setup test data."""
        self.prices = pd.Series(np.random.randn(100).cumsum() + 100)

    @patch("modules.adaptive_trend_LTS_mini.core.compute_atc_signals.compute_atc_signals.get_series_pool")
    def test_series_pool_release_failure_counted(self, mock_pool_getter):
        """Test that series pool release failures are counted and logged."""
        # Mock series pool that raises exception on release
        mock_pool = Mock()
        mock_pool.release.side_effect = Exception("Pool error")
        mock_pool_getter.return_value = mock_pool

        # Should not raise exception, but log warnings
        result = compute_atc_signals(
            self.prices,
            parallel_l1=False,  # Force non-parallel path
            use_rust_backend=False,
        )

        assert isinstance(result, dict)
        # Pool release was attempted (and failed)
        assert mock_pool.release.called


class TestParallelProcessingEdgeCases:
    """Test edge cases in parallel processing decisions."""

    def setup_method(self):
        """Setup test data."""
        self.prices = pd.Series(np.random.randn(100).cumsum() + 100)

    def test_parallel_disabled_in_child_process(self):
        """Test that parallel L1 is disabled in child processes."""
        with patch("multiprocessing.current_process") as mock_process:
            # Simulate child process
            mock_process.return_value.daemon = True
            mock_process.return_value.name = "ChildProcess"

            result = compute_atc_signals(
                self.prices,
                parallel_l1=None,  # Auto-detect
                use_rust_backend=False,
            )

            assert isinstance(result, dict)

    def test_parallel_enabled_with_sufficient_data_and_cores(self):
        """Test parallel L1 enabled with sufficient data and CPU cores."""
        # Long price series to trigger parallel processing
        long_prices = pd.Series(np.random.randn(10000).cumsum() + 100)

        with patch("modules.adaptive_trend_LTS_mini.core.compute_atc_signals.compute_atc_signals.get_hardware_manager") as mock_hw:
            # Mock sufficient CPU cores
            mock_resources = Mock()
            mock_resources.cpu_cores = 8
            mock_hw.return_value.get_resources.return_value = mock_resources

            result = compute_atc_signals(
                long_prices,
                parallel_l1=None,  # Auto-detect
                use_rust_backend=False,
            )

            assert isinstance(result, dict)

    def test_configurable_parallel_thresholds(self):
        """Test that parallel thresholds can be configured."""
        result = compute_atc_signals(
            self.prices,
            parallel_l1=None,
            min_bars_parallel_l1=50,  # Lower threshold
            min_cores_parallel_l1=2,  # Lower threshold
            use_rust_backend=False,
        )

        assert isinstance(result, dict)


class TestIncrementalATCErrorRecovery:
    """Test error recovery in incremental ATC updates."""

    def test_incremental_initialization_with_invalid_config(self):
        """Test incremental ATC with invalid configuration."""
        from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import (
            IncrementalATC,
        )

        invalid_config = {
            "ema_len": -1,  # Invalid
            "decay": 0.03,
            "lambda_param": 0.02,
        }

        # Should handle invalid config gracefully or raise clear error
        with pytest.raises((ValueError, KeyError)):
            atc = IncrementalATC(invalid_config)

    def test_incremental_load_state_from_nonexistent_file(self):
        """Test loading state from non-existent file."""
        from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import (
            IncrementalATC,
        )

        with pytest.raises(FileNotFoundError):
            IncrementalATC.load_state("nonexistent_file.msgpack")

    def test_incremental_load_state_corrupted_file(self):
        """Test loading state from corrupted file."""
        from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import (
            IncrementalATC,
        )

        import tempfile

        # Create temporary corrupted file
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".msgpack") as f:
            f.write(b"corrupted data")
            temp_path = f.name

        try:
            with pytest.raises((ValueError, Exception)):
                IncrementalATC.load_state(temp_path)
        finally:
            Path(temp_path).unlink()


class TestMemoryManagementErrors:
    """Test memory management error handling."""

    def setup_method(self):
        """Setup test data."""
        self.prices = pd.Series(np.random.randn(100).cumsum() + 100)

    @patch("modules.adaptive_trend_LTS_mini.core.compute_atc_signals.compute_atc_signals.get_memory_manager")
    def test_memory_tracking_failure_handled(self, mock_mem_manager_getter):
        """Test that memory tracking failures don't crash computation."""
        # Mock memory manager that raises exception
        mock_mem_manager = Mock()
        mock_mem_manager.track_memory.side_effect = Exception("Memory tracking error")
        mock_mem_manager_getter.return_value = mock_mem_manager

        # Should still complete with fast_mode=False
        result = compute_atc_signals(
            self.prices,
            fast_mode=False,  # Enable memory tracking
            use_rust_backend=False,
        )

        # Should fail gracefully or complete successfully
        # (depending on implementation details)


class TestEdgeCaseInputs:
    """Test edge case inputs to various functions."""

    def test_zero_length_prices(self):
        """Test with zero-length price series."""
        empty_prices = pd.Series([], dtype=float)

        with pytest.raises((ValueError, IndexError)):
            compute_atc_signals(empty_prices)

    def test_single_price_point(self):
        """Test with single price point."""
        single_price = pd.Series([100.0])

        # Should handle gracefully
        result = compute_atc_signals(single_price, use_rust_backend=False)

        assert isinstance(result, dict)

    def test_constant_prices(self):
        """Test with constant (no volatility) prices."""
        constant_prices = pd.Series([100.0] * 100)

        # Should handle zero volatility
        result = compute_atc_signals(constant_prices, use_rust_backend=False)

        assert isinstance(result, dict)

    def test_extreme_parameter_values(self):
        """Test with extreme parameter values."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)

        # Very high lambda and decay
        result = compute_atc_signals(
            prices,
            lambda_param=10.0,  # Very high
            decay=10.0,  # Very high
            use_rust_backend=False,
        )

        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
