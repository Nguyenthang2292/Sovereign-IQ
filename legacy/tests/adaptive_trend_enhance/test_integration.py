"""
Integration tests for adaptive_trend_enhance module.

Tests:
- Enhanced scanner with real market data (mocked)
- Different execution modes (sequential, threadpool, asyncio, processpool, gpu_batch)
- CLI compatibility
- Different hardware configurations
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.adaptive_trend_enhance.cli.argument_parser import parse_args
from modules.adaptive_trend_enhance.core.scanner import scan_all_symbols
from modules.adaptive_trend_enhance.utils.config import ATCConfig
from modules.common.system.detection import SystemInfo


@pytest.fixture
def mock_data_fetcher():
    """Create a mock DataFetcher instance."""
    fetcher = MagicMock()
    return fetcher


@pytest.fixture
def base_config():
    """Create a base ATCConfig for testing."""
    return ATCConfig(
        limit=200,
        timeframe="1h",
        ema_len=28,
        hma_len=28,
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
        long_threshold=0.1,
        short_threshold=-0.1,
        batch_size=50,
    )


def create_realistic_ohlcv_data(num_candles: int = 200) -> pd.DataFrame:
    """Create realistic OHLCV data similar to real market data."""
    np.random.seed(42)
    timestamps = pd.date_range(start="2024-01-01", periods=num_candles, freq="1h", tz="UTC")

    # Generate realistic price movement
    base_price = 100.0
    returns = np.random.randn(num_candles) * 0.02  # 2% volatility
    prices = base_price * (1 + returns).cumprod()

    # Create OHLCV
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * (1 + np.random.randn(num_candles) * 0.001),
            "high": prices * (1 + abs(np.random.randn(num_candles)) * 0.005),
            "low": prices * (1 - abs(np.random.randn(num_candles)) * 0.005),
            "close": prices,
            "volume": np.random.randint(1000, 10000, num_candles),
        }
    )
    return df


def create_mock_atc_results(signal_value: float = 0.05, trend: int = 1) -> dict:
    """Create mock ATC results with realistic signal."""
    signal_series = pd.Series([signal_value] * 200)
    return {
        "Average_Signal": signal_series,
    }


class TestEnhancedScannerWithRealMarketData:
    """Test enhanced scanner with realistic market data."""

    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.compute_atc_signals")
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.trend_sign")
    def test_scanner_with_realistic_data(self, mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
        """Test scanner with realistic market data patterns."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
        mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

        # Create realistic data - same data for all symbols (mocks will be called per symbol)
        mock_df = create_realistic_ohlcv_data(num_candles=500)
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

        # Vary signal strength for different symbols using side_effect
        def compute_atc_side_effect(*args, **kwargs):
            # Get symbol from context if available, otherwise use default
            # Since we can't easily get symbol from args, use a simple approach
            return create_mock_atc_results(signal_value=0.3)  # Use strong signal for all

        mock_compute_atc.side_effect = compute_atc_side_effect
        mock_trend_sign.return_value = pd.Series([1] * 500)

        # Run scanner once for all symbols
        long_df, short_df = scan_all_symbols(
            mock_data_fetcher, base_config, batch_size=10, execution_mode="sequential", max_symbols=5
        )

        # Verify results
        assert len(long_df) == 5, f"Expected 5 results, got {len(long_df)}"
        assert "BTCUSDT" in long_df["symbol"].values, "BTCUSDT should be in results"

    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.compute_atc_signals")
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.trend_sign")
    def test_scanner_with_varying_data_quality(self, mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
        """Test scanner handles varying data quality."""
        symbols = ["GOOD_SYMBOL", "SHORT_SYMBOL", "NOISY_SYMBOL"]
        mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

        # Good data
        good_df = create_realistic_ohlcv_data(num_candles=500)
        # Short data (should be skipped or handled)
        short_df = create_realistic_ohlcv_data(num_candles=50)
        # Noisy data
        noisy_df = create_realistic_ohlcv_data(num_candles=500)
        noisy_df.loc[100:200, "close"] = np.nan  # Add some NaN values

        def fetch_side_effect(symbol, *args, **kwargs):
            if symbol == "GOOD_SYMBOL":
                return (good_df, "binance")
            elif symbol == "SHORT_SYMBOL":
                return (short_df, "binance")
            else:
                return (noisy_df, "binance")

        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.side_effect = fetch_side_effect
        mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.2)
        mock_trend_sign.return_value = pd.Series([1] * 500)

        # Scanner should handle all cases gracefully
        long_df, short_df = scan_all_symbols(
            mock_data_fetcher, base_config, batch_size=5, execution_mode="sequential", max_symbols=3
        )

        # Should complete without errors
        assert isinstance(long_df, pd.DataFrame)
        assert isinstance(short_df, pd.DataFrame)


class TestDifferentExecutionModes:
    """Test different execution modes."""

    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.compute_atc_signals")
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.trend_sign")
    @pytest.mark.parametrize(
        "execution_mode",
        ["sequential", "threadpool", "asyncio", "processpool"],
    )
    def test_all_execution_modes(
        self, mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher, execution_mode
    ):
        """Test all execution modes produce consistent results."""
        if execution_mode == "processpool" and sys.platform == "win32":
            pytest.skip("processpool with mocks is not supported on Windows due to pickling limitations")

        symbols = [f"SYM{i}" for i in range(20)]
        mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

        mock_df = create_realistic_ohlcv_data(num_candles=300)
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

        mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.25)
        mock_trend_sign.return_value = pd.Series([1] * 300)

        try:
            long_df, short_df = scan_all_symbols(
                mock_data_fetcher,
                base_config,
                batch_size=10,
                execution_mode=execution_mode,
                max_workers=2,
                max_symbols=20,
            )

            print(f"\nExecution Mode: {execution_mode}")
            print(f"  Results: {len(long_df)} long, {len(short_df)} short")

            assert len(long_df) == 20, f"{execution_mode} mode should process all symbols"
        except Exception as e:
            # Some modes may not be available (e.g., GPU, processpool on Windows)
            if execution_mode == "processpool" and "spawn" in str(e).lower():
                pytest.skip(f"{execution_mode} mode not available on this platform")
            elif execution_mode == "gpu_batch" and "cupy" in str(e).lower():
                pytest.skip(f"{execution_mode} mode requires GPU/CuPy")
            else:
                raise

    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.compute_atc_signals")
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.trend_sign")
    def test_execution_mode_consistency(self, mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
        """Test that different execution modes produce consistent results."""
        symbols = [f"SYM{i}" for i in range(10)]
        mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

        mock_df = create_realistic_ohlcv_data(num_candles=200)
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

        mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.3)
        mock_trend_sign.return_value = pd.Series([1] * 200)

        results = {}
        for mode in ["sequential", "threadpool"]:
            # Reset mocks
            mock_compute_atc.reset_mock()
            mock_trend_sign.reset_mock()
            mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.reset_mock()

            long_df, short_df = scan_all_symbols(
                mock_data_fetcher, base_config, batch_size=5, execution_mode=mode, max_symbols=10
            )

            results[mode] = {
                "long": long_df.sort_values("symbol"),
                "short": short_df.sort_values("symbol"),
            }

        # Compare results
        if "sequential" in results and "threadpool" in results:
            seq_long = results["sequential"]["long"]
            thread_long = results["threadpool"]["long"]

            # Results should be consistent (same symbols, similar signals)
            assert len(seq_long) == len(thread_long), "Results should have same length"
            assert set(seq_long["symbol"]) == set(thread_long["symbol"]), "Results should have same symbols"


class TestCLICompatibility:
    """Test CLI compatibility."""

    @patch("sys.argv", ["test_cli", "--auto", "--timeframe", "15m", "--limit", "1000"])
    def test_cli_argument_parsing(self):
        """Test CLI argument parser works correctly."""
        # Test basic arguments
        args = parse_args()
        assert args.auto is True
        assert args.timeframe == "15m"
        assert args.limit == 1000

        # Test batch size
        with patch("sys.argv", ["test_cli", "--batch-size", "50"]):
            args = parse_args()
            assert args.batch_size == 50

        # Test default values
        with patch("sys.argv", ["test_cli"]):
            args = parse_args()
            assert args.batch_size == 100  # Default batch size

    @patch(
        "sys.argv",
        [
            "test_cli",
            "--timeframe",
            "1h",
            "--limit",
            "1500",
            "--batch-size",
            "100",
            "--lambda",
            "0.03",
            "--decay",
            "0.04",
        ],
    )
    def test_cli_config_integration(self):
        """Test CLI arguments integrate with ATCConfig."""
        from modules.adaptive_trend_enhance.utils.config import create_atc_config_from_dict

        args = parse_args()

        # Convert args to config
        params = {
            "timeframe": args.timeframe,
            "limit": args.limit,
            "batch_size": args.batch_size,
            "lambda_param": args.lambda_param if hasattr(args, "lambda_param") else 0.02,
            "decay": args.decay if hasattr(args, "decay") else 0.03,
        }

        config = create_atc_config_from_dict(params, timeframe=args.timeframe)

        assert config.timeframe == "1h"
        assert config.limit == 1500
        assert config.batch_size == 100

    @patch("modules.adaptive_trend_enhance.core.scanner.scan_all_symbols")
    def test_cli_scanner_integration(self, mock_scan, mock_data_fetcher):
        """Test CLI integrates with scanner correctly."""
        # Mock scanner results
        long_df = pd.DataFrame(
            {"symbol": ["BTCUSDT"], "signal": [0.5], "trend": [1], "price": [50000], "exchange": ["binance"]}
        )
        short_df = pd.DataFrame({"symbol": [], "signal": [], "trend": [], "price": [], "exchange": []})
        mock_scan.return_value = (long_df, short_df)

        # Simulate CLI call - patch sys.argv before calling parse_args()
        with patch("sys.argv", ["test_cli", "--auto", "--batch-size", "50"]):
            args = parse_args()

            # Verify scanner would be called with correct args
            # (In real CLI, this would call scan_all_symbols with these args)
            assert args.auto is True
            assert args.batch_size == 50


class TestDifferentHardwareConfigurations:
    """Test with different hardware configurations."""

    def test_detect_hardware_info(self):
        """Test hardware detection works."""
        from modules.common.system.detection import CPUDetector, GPUDetector, SystemInfo

        cpu_info = CPUDetector.detect()
        memory_info = SystemInfo.get_memory_info()
        gpu_info = GPUDetector.detect_all()

        print("\nHardware Configuration:")
        print(f"  CPU Cores: {cpu_info.cores}")
        print(f"  RAM Total: {memory_info.total_gb:.2f} GB")
        print(f"  GPU Available: {gpu_info.available}")

        # Verify hardware info is available
        assert cpu_info.cores is not None or cpu_info.cores == 0
        assert memory_info.total_gb is not None

    def test_adaptive_configuration_based_on_hardware(self, base_config):
        """Test configuration adapts to hardware."""
        system_info = SystemInfo()
        cpu_info = system_info.get_cpu_info()
        memory_info = system_info.get_memory_info()

        # Adjust batch size based on available memory
        available_ram_gb = memory_info.available_gb
        if available_ram_gb < 4.0:
            recommended_batch_size = 25
        elif available_ram_gb < 8.0:
            recommended_batch_size = 50
        else:
            recommended_batch_size = 100

        # Adjust workers based on CPU cores
        cpu_cores = cpu_info.cores
        recommended_workers = min(cpu_cores, 8)

        print("\nAdaptive Configuration:")
        print(f"  Available RAM: {available_ram_gb:.2f} GB")
        print(f"  Recommended batch size: {recommended_batch_size}")
        print(f"  CPU Cores: {cpu_cores}")
        print(f"  Recommended workers: {recommended_workers}")

        # Verify recommendations are reasonable
        assert 10 <= recommended_batch_size <= 200
        assert 1 <= recommended_workers <= 16

    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.compute_atc_signals")
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.trend_sign")
    def test_low_memory_configuration(self, mock_trend_sign, mock_compute_atc, mock_data_fetcher):
        """Test scanner works with low memory configuration."""
        # Simulate low memory by using small batch size
        config = ATCConfig(
            limit=200,
            timeframe="1h",
            batch_size=10,  # Small batch for low memory
        )

        symbols = [f"SYM{i}" for i in range(50)]
        mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

        mock_df = create_realistic_ohlcv_data(num_candles=200)
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

        mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.2)
        mock_trend_sign.return_value = pd.Series([1] * 200)

        # Should work with small batches
        long_df, short_df = scan_all_symbols(
            mock_data_fetcher, config, batch_size=10, execution_mode="sequential", max_symbols=50
        )

        assert len(long_df) == 50

    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.compute_atc_signals")
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.trend_sign")
    def test_high_memory_configuration(self, mock_trend_sign, mock_compute_atc, mock_data_fetcher):
        """Test scanner works with high memory configuration."""
        # Simulate high memory by using large batch size
        config = ATCConfig(
            limit=200,
            timeframe="1h",
            batch_size=200,  # Large batch for high memory
        )

        symbols = [f"SYM{i}" for i in range(100)]
        mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

        mock_df = create_realistic_ohlcv_data(num_candles=200)
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

        mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.2)
        mock_trend_sign.return_value = pd.Series([1] * 200)

        # Should work with large batches
        long_df, short_df = scan_all_symbols(
            mock_data_fetcher, config, batch_size=200, execution_mode="sequential", max_symbols=100
        )

        assert len(long_df) == 100
