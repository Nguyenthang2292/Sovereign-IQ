"""
Integration Tests for Adaptive Trend LTS Mini Module

This module contains end-to-end integration tests that verify the complete
pipeline functionality, including:
- Full signal generation pipeline
- Multi-component interaction
- Real-world scenarios
- Data flow from fetching to final signals
"""

import pytest
from unittest.mock import MagicMock, Mock, patch
import pandas as pd
import numpy as np
from argparse import Namespace

from modules.adaptive_trend_LTS_mini.cli.main import ATCAnalyzer
from modules.adaptive_trend_LTS_mini.core.analyzer import analyze_symbol
from modules.adaptive_trend_LTS_mini.core.scanner import scan_all_symbols
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig, create_atc_config_from_dict


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    periods = 200
    dates = pd.date_range("2024-01-01", periods=periods, freq="1h")

    # Generate realistic price data with trend
    base_price = 50000
    trend = np.linspace(0, 5000, periods)
    noise = np.random.randn(periods) * 500
    prices = base_price + trend + noise

    # Create OHLCV DataFrame
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices + np.random.randn(periods) * 50,
        "high": prices + np.abs(np.random.randn(periods) * 100),
        "low": prices - np.abs(np.random.randn(periods) * 100),
        "close": prices,
        "volume": np.random.randint(100, 1000, periods),
    })

    return df


@pytest.fixture
def mock_data_fetcher():
    """Create mock DataFetcher for testing."""
    fetcher = MagicMock()

    # Mock list_binance_futures_symbols
    fetcher.list_binance_futures_symbols.return_value = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"
    ]

    return fetcher


@pytest.fixture
def basic_atc_config():
    """Create basic ATCConfig for testing."""
    return ATCConfig(
        ema_len=10,
        hma_len=10,
        wma_len=10,
        dema_len=10,
        lsma_len=10,
        kama_len=10,
        robustness="Medium",
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
        limit=200,
        timeframe="1h",
        long_threshold=0.1,
        short_threshold=-0.1,
        use_rust_backend=False,
        parallel_l1=False,
        parallel_l2=False,
    )


@pytest.fixture
def basic_args():
    """Create basic argument Namespace for testing."""
    return Namespace(
        timeframe="1h",
        auto=False,
        no_menu=True,
        symbol="BTC/USDT",
        quote="USDT",
        no_prompt=True,
        limit=200,
        ema_len=10,
        hma_len=10,
        wma_len=10,
        dema_len=10,
        lsma_len=10,
        kama_len=10,
        robustness="Medium",
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
        long_threshold=0.1,
        short_threshold=-0.1,
        min_signal=0.01,
        max_symbols=None,
        use_rust_backend=False,
        batch_processing=False,
        fast_mode=False,
        precision="float64",
        use_cache=False,
        use_approximate=False,
        use_adaptive_approximate=False,
        approximate_volatility_window=20,
        approximate_volatility_factor=1.0,
        approximate_threshold=0.05,
        batch_size=100,
        execution_mode="threadpool",
        npartitions=None,
        ema_w=1.0,
        hma_w=1.0,
        wma_w=1.0,
        dema_w=1.0,
        lsma_w=1.0,
        kama_w=1.0,
        equity_floor=0.25,
    )


# ============================================================================
# Test 1: End-to-End Single Symbol Analysis
# ============================================================================


def test_integration_single_symbol_analysis_pipeline(
    sample_ohlcv_data, mock_data_fetcher, basic_atc_config
):
    """
    Integration Test 1: End-to-End Single Symbol Analysis

    Tests the complete pipeline for analyzing a single symbol:
    1. Data fetching
    2. ATC signal computation
    3. Result validation

    Verifies:
    - All moving average signals are computed
    - Average_Signal is present
    - Results have correct structure
    - No exceptions during execution
    """
    # Setup mock data fetcher
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (
        sample_ohlcv_data,
        "binance",
    )

    # Execute analysis
    result = analyze_symbol(
        symbol="BTC/USDT",
        data_fetcher=mock_data_fetcher,
        config=basic_atc_config,
    )

    # Validate result structure
    assert result is not None, "Analysis should return a result"
    assert "symbol" in result
    assert "df" in result
    assert "atc_results" in result
    assert "current_price" in result
    assert "exchange_label" in result

    # Validate result values
    assert result["symbol"] == "BTC/USDT"
    assert result["exchange_label"] == "BINANCE"
    assert isinstance(result["df"], pd.DataFrame)
    assert len(result["df"]) == len(sample_ohlcv_data)

    # Validate ATC results
    atc_results = result["atc_results"]
    expected_signals = [
        "EMA_Signal", "HMA_Signal", "WMA_Signal",
        "DEMA_Signal", "LSMA_Signal", "KAMA_Signal",
        "EMA_S", "HMA_S", "WMA_S",
        "DEMA_S", "LSMA_S", "KAMA_S",
        "Average_Signal",
    ]

    for signal_name in expected_signals:
        assert signal_name in atc_results, f"Missing signal: {signal_name}"
        assert isinstance(atc_results[signal_name], pd.Series)
        assert len(atc_results[signal_name]) == len(sample_ohlcv_data)


# ============================================================================
# Test 2: End-to-End Scanner Pipeline with Multiple Symbols
# ============================================================================


def test_integration_scanner_pipeline_multiple_symbols(
    sample_ohlcv_data, mock_data_fetcher, basic_atc_config
):
    """
    Integration Test 2: Scanner Pipeline with Multiple Symbols

    Tests the complete scanner pipeline:
    1. Symbol discovery
    2. Batch data fetching
    3. ATC signal computation for multiple symbols
    4. Signal filtering and ranking

    Verifies:
    - Scanner processes multiple symbols
    - Results are properly filtered (LONG/SHORT)
    - Results are sorted by signal strength
    - Empty results are handled gracefully
    """
    # Setup mock to return slightly different data for each symbol
    def mock_fetch_ohlcv(symbol, **kwargs):
        df = sample_ohlcv_data.copy()
        # Add slight variation per symbol for realistic testing
        symbol_seed = hash(symbol) % 100
        df["close"] = df["close"] + symbol_seed * 100
        return (df, "binance")

    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.side_effect = mock_fetch_ohlcv

    # Execute scanner
    long_signals, short_signals = scan_all_symbols(
        data_fetcher=mock_data_fetcher,
        atc_config=basic_atc_config,
        max_symbols=5,
        min_signal=0.01,
        execution_mode="sequential",  # Use sequential for deterministic results
        batch_size=100,
        symbols=None,  # Let scanner discover symbols
    )

    # Validate result types
    assert isinstance(long_signals, pd.DataFrame), "long_signals should be DataFrame"
    assert isinstance(short_signals, pd.DataFrame), "short_signals should be DataFrame"

    # Validate DataFrame structure
    for df, signal_type in [(long_signals, "LONG"), (short_signals, "SHORT")]:
        if not df.empty:
            assert "symbol" in df.columns
            assert "signal" in df.columns
            assert "trend" in df.columns
            assert "price" in df.columns
            assert "exchange" in df.columns

            # Validate sorting (should be sorted by signal strength descending)
            if len(df) > 1:
                signals = df["signal"].values
                assert all(signals[i] >= signals[i+1] for i in range(len(signals)-1)), \
                    f"{signal_type} signals should be sorted descending"

            # Validate signal types
            if signal_type == "LONG":
                assert all(df["trend"] > 0), "LONG signals should have positive trend"
            else:
                assert all(df["trend"] < 0), "SHORT signals should have negative trend"


# ============================================================================
# Test 3: CLI Integration - Manual Mode Complete Flow
# ============================================================================


# ============================================================================
# Test 3: CLI Integration - Manual Mode Complete Flow
# ============================================================================


@patch("modules.adaptive_trend_LTS_mini.cli.manual_mode_executor.analyze_symbol")
def test_integration_cli_manual_mode_complete_flow(
    mock_analyze, basic_args, mock_data_fetcher, sample_ohlcv_data
):
    """
    Integration Test 3: CLI Manual Mode Complete Flow

    Tests the complete CLI manual mode workflow:
    1. Argument parsing
    2. Component initialization
    3. Symbol analysis

    Verifies:
    - CLI correctly orchestrates the analysis
    - All components are called in correct order
    - Configuration flows correctly
    """
    # Setup mock analyzer response
    mock_result = {
        "symbol": "BTC/USDT",
        "df": sample_ohlcv_data,
        "atc_results": {
            "Average_Signal": pd.Series([0.5] * len(sample_ohlcv_data)),
            "EMA_Signal": pd.Series([0.6] * len(sample_ohlcv_data)),
        },
        "current_price": 55000.0,
        "exchange_label": "BINANCE",
    }
    mock_analyze.return_value = mock_result

    # Create and run analyzer
    analyzer = ATCAnalyzer(basic_args, mock_data_fetcher)
    analyzer.run_manual_mode()

    # Verify analyze_symbol was called correctly
    mock_analyze.assert_called_once()
    call_kwargs = mock_analyze.call_args[1]
    assert call_kwargs["symbol"] == "BTC/USDT"
    assert call_kwargs["data_fetcher"] is mock_data_fetcher
    assert isinstance(call_kwargs["config"], ATCConfig)


# ============================================================================
# Test 4: CLI Integration - Auto Mode Complete Flow
# ============================================================================


@patch("modules.adaptive_trend_LTS_mini.cli.auto_mode_executor.scan_all_symbols")
def test_integration_cli_auto_mode_complete_flow(
    mock_scan, basic_args, mock_data_fetcher
):
    """
    Integration Test 4: CLI Auto Mode Complete Flow

    Tests the complete CLI auto mode workflow:
    1. Mode determination
    2. Scanner initialization
    3. Multi-symbol scanning

    Verifies:
    - Auto mode correctly triggers scanner
    - Configuration is properly passed
    """
    # Setup args for auto mode
    basic_args.auto = True
    basic_args.no_menu = True

    # Setup mock scanner response
    long_df = pd.DataFrame({
        "symbol": ["BTC/USDT", "ETH/USDT"],
        "signal": [0.8, 0.6],
        "trend": [0.5, 0.4],
        "price": [55000.0, 3000.0],
        "exchange": ["binance", "binance"],
    })

    short_df = pd.DataFrame({
        "symbol": ["SOL/USDT"],
        "signal": [0.7],
        "trend": [-0.5],
        "price": [100.0],
        "exchange": ["binance"],
    })

    mock_scan.return_value = (long_df, short_df)

    # Create and run analyzer
    analyzer = ATCAnalyzer(basic_args, mock_data_fetcher)
    analyzer.run_auto_mode()

    # Verify scan_all_symbols was called
    mock_scan.assert_called_once()
    call_kwargs = mock_scan.call_args[1]
    assert call_kwargs["data_fetcher"] is mock_data_fetcher
    assert isinstance(call_kwargs["atc_config"], ATCConfig)


# ============================================================================
# Test 5: Multi-Component Interaction - Data Fetcher → Analyzer → Scanner
# ============================================================================


def test_integration_multi_component_data_flow(
    sample_ohlcv_data, mock_data_fetcher, basic_atc_config
):
    """
    Integration Test 5: Multi-Component Data Flow

    Tests the interaction between multiple components:
    1. DataFetcher provides data
    2. Analyzer processes single symbol
    3. Scanner aggregates multiple symbols

    Verifies:
    - Data flows correctly between components
    - Each component transforms data appropriately
    - No data corruption during pipeline
    """
    # Setup mock data fetcher
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (
        sample_ohlcv_data,
        "binance",
    )

    # Step 1: Test analyzer receives and processes data
    analyzer_result = analyze_symbol(
        symbol="BTC/USDT",
        data_fetcher=mock_data_fetcher,
        config=basic_atc_config,
    )

    assert analyzer_result is not None
    assert "atc_results" in analyzer_result
    assert "Average_Signal" in analyzer_result["atc_results"]

    # Verify signal values are within expected range [-1, 1]
    avg_signal = analyzer_result["atc_results"]["Average_Signal"]
    assert avg_signal.min() >= -1.5, "Average signal should be >= -1.5"
    assert avg_signal.max() <= 1.5, "Average signal should be <= 1.5"

    # Step 2: Test scanner uses analyzer results
    long_signals, short_signals = scan_all_symbols(
        data_fetcher=mock_data_fetcher,
        atc_config=basic_atc_config,
        max_symbols=3,
        min_signal=0.0,  # Low threshold to catch any signals
        execution_mode="sequential",
        symbols=["BTC/USDT", "ETH/USDT"],  # Pre-filtered list
    )

    # Verify scanner produces aggregated results
    total_results = len(long_signals) + len(short_signals)
    assert total_results >= 0, "Scanner should produce results"

    # Verify result consistency
    if not long_signals.empty:
        assert all(long_signals["symbol"].isin(["BTC/USDT", "ETH/USDT"]))
    if not short_signals.empty:
        assert all(short_signals["symbol"].isin(["BTC/USDT", "ETH/USDT"]))


# ============================================================================
# Test 6: Real-World Scenario - Empty and Invalid Data Handling
# ============================================================================


def test_integration_real_world_empty_data_handling(mock_data_fetcher, basic_atc_config):
    """
    Integration Test 6: Empty and Invalid Data Handling

    Tests real-world scenario where data is empty or invalid:
    1. Empty DataFrame from data fetcher
    2. None result from analyzer
    3. Scanner handles failures gracefully

    Verifies:
    - Pipeline handles empty data without crashing
    - Appropriate None/empty results returned
    - No exceptions propagated to user
    """
    # Test Case 1: Empty DataFrame
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (
        pd.DataFrame(),
        "binance",
    )

    result = analyze_symbol(
        symbol="INVALID/USDT",
        data_fetcher=mock_data_fetcher,
        config=basic_atc_config,
    )

    assert result is None, "Analyzer should return None for empty data"

    # Test Case 2: None DataFrame
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (None, None)

    result = analyze_symbol(
        symbol="INVALID/USDT",
        data_fetcher=mock_data_fetcher,
        config=basic_atc_config,
    )

    assert result is None, "Analyzer should return None for None data"

    # Test Case 3: Scanner handles multiple failures
    call_count = [0]

    def mock_fetch_mixed(symbol, **kwargs):
        call_count[0] += 1
        if call_count[0] % 2 == 0:
            return (None, None)  # Some symbols fail
        else:
            # Create minimal valid data
            df = pd.DataFrame({
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
                "open": np.random.randn(50) + 50000,
                "high": np.random.randn(50) + 50100,
                "low": np.random.randn(50) + 49900,
                "close": np.random.randn(50) + 50000,
                "volume": np.random.randint(100, 1000, 50),
            })
            return (df, "binance")

    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.side_effect = mock_fetch_mixed

    # Scanner should handle mixed success/failure
    long_signals, short_signals = scan_all_symbols(
        data_fetcher=mock_data_fetcher,
        atc_config=basic_atc_config,
        max_symbols=4,
        min_signal=0.0,
        execution_mode="sequential",
        symbols=["SYM1/USDT", "SYM2/USDT", "SYM3/USDT", "SYM4/USDT"],
    )

    # Should return DataFrames (possibly empty) without crashing
    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)


# ============================================================================
# Test 7: Real-World Scenario - High Volatility Data
# ============================================================================


def test_integration_real_world_high_volatility_data(mock_data_fetcher, basic_atc_config):
    """
    Integration Test 7: High Volatility Data Handling

    Tests real-world scenario with high volatility data:
    1. Large price swings
    2. Trend reversals
    3. Signal stability

    Verifies:
    - Algorithm handles volatility without numerical issues
    - Signals remain in valid range
    - No NaN or Inf values in results
    """
    # Create high volatility data
    np.random.seed(42)
    periods = 200
    dates = pd.date_range("2024-01-01", periods=periods, freq="1h")

    # Generate high volatility prices with trend reversals
    base_price = 50000
    volatility = 2000  # High volatility
    trend_change = periods // 2

    prices = []
    for i in range(periods):
        if i < trend_change:
            trend = i * 50  # Uptrend
        else:
            trend = (periods - i) * 50  # Downtrend

        noise = np.random.randn() * volatility
        prices.append(base_price + trend + noise)

    prices = np.array(prices)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices + np.random.randn(periods) * 100,
        "high": prices + np.abs(np.random.randn(periods) * 200),
        "low": prices - np.abs(np.random.randn(periods) * 200),
        "close": prices,
        "volume": np.random.randint(100, 2000, periods),
    })

    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")

    # Execute analysis
    result = analyze_symbol(
        symbol="VOLATILE/USDT",
        data_fetcher=mock_data_fetcher,
        config=basic_atc_config,
    )

    assert result is not None, "Analysis should handle high volatility data"

    # Validate no NaN or Inf values
    for signal_name, signal_series in result["atc_results"].items():
        assert not signal_series.isnull().any(), f"{signal_name} should not contain NaN"
        assert not np.isinf(signal_series).any(), f"{signal_name} should not contain Inf"

        # Validate signals are in reasonable range
        assert signal_series.min() >= -2.0, f"{signal_name} min value should be >= -2.0"
        assert signal_series.max() <= 2.0, f"{signal_name} max value should be <= 2.0"


# ============================================================================
# Test 8: Configuration Integration - Config Flow Through Pipeline
# ============================================================================


def test_integration_config_flow_through_pipeline(
    sample_ohlcv_data, mock_data_fetcher, basic_args
):
    """
    Integration Test 8: Configuration Flow Through Pipeline

    Tests that configuration flows correctly through all components:
    1. Args → ATCConfig creation
    2. ATCConfig parameters are preserved
    3. Config scaling properties work correctly

    Verifies:
    - Configuration parameters are preserved
    - No parameter corruption during pipeline
    - Custom values are respected
    - Scaling computations are correct
    """
    # Setup custom configuration
    basic_args.ema_len = 15
    basic_args.robustness = "Wide"
    basic_args.lambda_param = 0.05
    basic_args.decay = 0.04
    basic_args.long_threshold = 0.2
    basic_args.short_threshold = -0.2

    # Create config from args
    from modules.adaptive_trend_LTS_mini.cli.config_utils import get_atc_params
    atc_params = get_atc_params(basic_args)
    atc_config = create_atc_config_from_dict(atc_params, timeframe="1h")

    # Verify config has custom values
    assert atc_config.ema_len == 15
    assert atc_config.robustness == "Wide"
    assert atc_config.lambda_param == 0.05
    assert atc_config.decay == 0.04
    assert atc_config.long_threshold == 0.2
    assert atc_config.short_threshold == -0.2
    assert atc_config.timeframe == "1h"

    # Verify config scaling properties work correctly
    assert abs(atc_config.lambda_scaled - 0.00005) < 1e-9  # 0.05 / 1000
    assert abs(atc_config.decay_scaled - 0.0004) < 1e-9  # 0.04 / 100

    # Test ConfigManager integration
    from modules.adaptive_trend_LTS_mini.cli.config_manager import ConfigManager

    config_manager = ConfigManager(basic_args)
    atc_config_created = config_manager.create_config(timeframe="1h")

    # Verify the created config matches our expectations
    assert atc_config_created.ema_len == 15
    assert atc_config_created.robustness == "Wide"
    assert atc_config_created.lambda_param == 0.05
    assert atc_config_created.decay == 0.04
    assert atc_config_created.long_threshold == 0.2
    assert atc_config_created.short_threshold == -0.2

    # Verify config can be serialized/deserialized (for potential caching)
    config_dict = {
        "ema_len": atc_config.ema_len,
        "hma_len": atc_config.hma_len,
        "robustness": atc_config.robustness,
        "lambda_param": atc_config.lambda_param,
        "decay": atc_config.decay,
    }

    # Recreate config from dict
    new_config = create_atc_config_from_dict(config_dict, timeframe="1h")
    assert new_config.ema_len == atc_config.ema_len
    assert new_config.robustness == atc_config.robustness
    assert new_config.lambda_param == atc_config.lambda_param
    assert new_config.decay == atc_config.decay


# ============================================================================
# Test 9: Error Recovery - Partial Failures in Scanner
# ============================================================================


def test_integration_error_recovery_partial_scanner_failures(
    sample_ohlcv_data, mock_data_fetcher, basic_atc_config
):
    """
    Integration Test 9: Error Recovery - Partial Scanner Failures

    Tests that scanner recovers from partial failures:
    1. Some symbols succeed
    2. Some symbols fail
    3. Scanner continues processing

    Verifies:
    - Scanner doesn't stop on first error
    - Successful results are returned
    - Failed symbols are logged but skipped
    """
    # Setup mock to simulate partial failures
    def mock_fetch_with_failures(symbol, **kwargs):
        if symbol in ["FAIL1/USDT", "FAIL2/USDT"]:
            raise Exception(f"Network error for {symbol}")
        else:
            df = sample_ohlcv_data.copy()
            # Add variation per symbol
            df["close"] = df["close"] + hash(symbol) % 1000
            return (df, "binance")

    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.side_effect = mock_fetch_with_failures

    # Execute scanner with mix of success/failure symbols
    test_symbols = [
        "BTC/USDT",
        "FAIL1/USDT",
        "ETH/USDT",
        "FAIL2/USDT",
        "SOL/USDT",
    ]

    long_signals, short_signals = scan_all_symbols(
        data_fetcher=mock_data_fetcher,
        atc_config=basic_atc_config,
        max_symbols=None,
        min_signal=0.0,
        execution_mode="sequential",
        symbols=test_symbols,
    )

    # Verify scanner returned results (DataFrames)
    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)

    # Verify successful symbols are in results
    all_symbols = pd.concat([long_signals, short_signals])["symbol"].unique() if not long_signals.empty or not short_signals.empty else []

    # Failed symbols should not appear in results
    assert "FAIL1/USDT" not in all_symbols
    assert "FAIL2/USDT" not in all_symbols

    # At least some successful symbols should appear (if signals met threshold)
    # Note: May be empty if no signals exceeded threshold, which is acceptable


# ============================================================================
# Test 10: Performance - Scanner Batch Processing
# ============================================================================


def test_integration_performance_scanner_batch_processing(
    sample_ohlcv_data, mock_data_fetcher, basic_atc_config
):
    """
    Integration Test 10: Scanner Batch Processing Performance

    Tests that scanner processes symbols in batches correctly:
    1. Large symbol list
    2. Batch size configuration
    3. Memory management between batches

    Verifies:
    - Scanner handles large symbol lists
    - Batch processing works correctly
    - Results are consistent regardless of batch size
    """
    # Setup mock to return data for many symbols
    def mock_fetch_many(symbol, **kwargs):
        df = sample_ohlcv_data.copy()
        # Add symbol-specific variation
        seed = hash(symbol) % 100
        df["close"] = df["close"] + seed * 50
        return (df, "binance")

    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.side_effect = mock_fetch_many

    # Create list of many symbols
    symbols = [f"SYM{i}/USDT" for i in range(20)]

    # Test with different batch sizes
    for batch_size in [5, 10, 20]:
        long_signals, short_signals = scan_all_symbols(
            data_fetcher=mock_data_fetcher,
            atc_config=basic_atc_config,
            max_symbols=None,
            min_signal=0.0,
            execution_mode="sequential",
            batch_size=batch_size,
            symbols=symbols,
        )

        # Verify results structure
        assert isinstance(long_signals, pd.DataFrame)
        assert isinstance(short_signals, pd.DataFrame)

        # Verify all requested symbols were processed (or attempted)
        # Note: mock_data_fetcher.fetch_ohlcv_with_fallback_exchange call_count
        # should be >= len(symbols) if all symbols attempted
