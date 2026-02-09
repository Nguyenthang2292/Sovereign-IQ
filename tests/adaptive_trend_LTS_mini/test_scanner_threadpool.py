"""
Tests for ATC scanner threadpool: single executor, error handling, cancellation.

Covers Bottleneck #5 fixes: single ThreadPoolExecutor for entire scan,
gc.collect() only at end, and worker error handling / KeyboardInterrupt.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from modules.adaptive_trend_LTS_mini.core.scanner import scan_all_symbols
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig


@pytest.fixture
def mock_atc_config():
    return ATCConfig(
        timeframe="1h",
        limit=100,
        ema_len=10,
        hma_len=10,
        wma_len=10,
        dema_len=10,
        lsma_len=10,
        kama_len=10,
        robustness="Medium",
        lambda_param=0.1,
        decay=0.1,
        cutout=0,
        batch_size=50,
    )


@pytest.fixture
def mock_data_fetcher():
    fetcher = Mock()
    periods = 100
    dates = pd.date_range("2023-01-01", periods=periods, freq="h")
    import numpy as np
    prices = 100 + np.cumsum(np.random.randn(periods) * 0.5)
    prices = np.maximum(prices, 10.0)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.uniform(1000, 10000, periods),
        },
        index=dates,
    )
    fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")
    fetcher.list_binance_futures_symbols.return_value = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    ]
    fetcher.should_stop.return_value = False
    return fetcher


# ==================== SINGLE EXECUTOR (Bottleneck #5) ====================


def test_threadpool_uses_single_executor_for_entire_scan(
    mock_data_fetcher, mock_atc_config
):
    """ThreadPoolExecutor is created once for the entire scan, not per batch."""
    create_count = 0
    real_executor = None

    with patch(
        "modules.adaptive_trend_LTS_mini.core.scanner.threadpool.ThreadPoolExecutor"
    ) as mock_tp_class:
        from concurrent.futures import ThreadPoolExecutor as RealTP

        def capture_create(*args, **kwargs):
            nonlocal create_count
            create_count += 1
            return RealTP(*args, **kwargs)

        mock_tp_class.side_effect = capture_create

        with patch(
            "modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager"
        ) as mock_hw:
            with patch(
                "modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager"
            ) as mock_mem:
                mock_hw.return_value.get_optimal_workload_config.return_value = Mock(
                    num_threads=4, num_processes=2
                )
                mock_mem.return_value.safe_memory_operation.return_value.__enter__ = Mock()
                mock_mem.return_value.safe_memory_operation.return_value.__exit__ = Mock()
                mock_mem.return_value.log_memory_stats = Mock()

                scan_all_symbols(
                    mock_data_fetcher,
                    mock_atc_config,
                    symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
                    execution_mode="threadpool",
                    batch_size=2,
                )

    assert create_count == 1, "ThreadPoolExecutor should be instantiated once for entire scan"


# ==================== ERROR HANDLING IN WORKERS ====================


def test_threadpool_worker_exception_continues_other_symbols(
    mock_data_fetcher, mock_atc_config
):
    """When one symbol raises in _process_symbol, others complete and errors are counted."""
    invoked = []

    def patched_process(symbol, data_fetcher, atc_config, min_signal):
        invoked.append(symbol)
        if symbol in ("ETH/USDT", "ETHUSDT"):
            raise RuntimeError("Simulated worker failure")
        return {
            "symbol": symbol,
            "signal": 0.1,
            "trend": 1,
            "price": 50000.0,
            "exchange": "binance",
        }

    with patch(
        "modules.adaptive_trend_LTS_mini.core.scanner.threadpool._process_symbol",
        side_effect=patched_process,
    ):
        with patch(
            "modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager"
        ) as mock_hw:
            with patch(
                "modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager"
            ) as mock_mem:
                mock_hw.return_value.get_optimal_workload_config.return_value = Mock(
                    num_threads=4, num_processes=2
                )
                mock_mem.return_value.safe_memory_operation.return_value.__enter__ = Mock()
                mock_mem.return_value.safe_memory_operation.return_value.__exit__ = Mock()
                mock_mem.return_value.log_memory_stats = Mock()

                long_df, short_df = scan_all_symbols(
                    mock_data_fetcher,
                    mock_atc_config,
                    symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                    execution_mode="threadpool",
                )

    assert any("ETH" in s for s in invoked), "Patched _process_symbol should have been called for ETH"
    assert len(long_df) == 2, f"Expected 2 results when one symbol raises, got {len(long_df)}: {long_df}"
    symbols_in_df = long_df["symbol"].astype(str).str.upper().tolist()
    assert "ETH/USDT" not in symbols_in_df and "ETHUSDT" not in symbols_in_df
    assert "BTC/USDT" in symbols_in_df or "BTCUSDT" in symbols_in_df
    assert "SOL/USDT" in symbols_in_df or "SOLUSDT" in symbols_in_df


def test_threadpool_gc_called_once_after_scan(
    mock_data_fetcher, mock_atc_config
):
    """gc.collect() is invoked once after the full scan, not per batch."""
    gc_calls = []

    with patch(
        "modules.adaptive_trend_LTS_mini.core.scanner.threadpool.gc.collect",
        side_effect=lambda: gc_calls.append(1),
    ):
        with patch(
            "modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager"
        ) as mock_hw:
            with patch(
                "modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager"
            ) as mock_mem:
                mock_hw.return_value.get_optimal_workload_config.return_value = Mock(
                    num_threads=4, num_processes=2
                )
                mock_mem.return_value.safe_memory_operation.return_value.__enter__ = Mock()
                mock_mem.return_value.safe_memory_operation.return_value.__exit__ = Mock()
                mock_mem.return_value.log_memory_stats = Mock()

                scan_all_symbols(
                    mock_data_fetcher,
                    mock_atc_config,
                    symbols=["BTC/USDT", "ETH/USDT"],
                    execution_mode="threadpool",
                    batch_size=1,
                )

    assert len(gc_calls) == 1, "gc.collect() should be called once after scan"


def test_threadpool_keyboard_interrupt_handled_cleanly(
    mock_data_fetcher, mock_atc_config
):
    """On KeyboardInterrupt during as_completed, threadpool catches it and returns without crashing."""
    from concurrent.futures import as_completed as real_as_completed

    def raise_after_first(*args, **kwargs):
        it = real_as_completed(*args, **kwargs)
        next(it)
        raise KeyboardInterrupt()

    with patch(
        "modules.adaptive_trend_LTS_mini.core.scanner.threadpool.as_completed",
        side_effect=raise_after_first,
    ):
        with patch(
            "modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager"
        ) as mock_hw:
            with patch(
                "modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager"
            ) as mock_mem:
                mock_hw.return_value.get_optimal_workload_config.return_value = Mock(
                    num_threads=4, num_processes=2
                )
                mock_mem.return_value.safe_memory_operation.return_value.__enter__ = Mock()
                mock_mem.return_value.safe_memory_operation.return_value.__exit__ = Mock()
                mock_mem.return_value.log_memory_stats = Mock()

                long_df, short_df = scan_all_symbols(
                    mock_data_fetcher,
                    mock_atc_config,
                    symbols=["BTC/USDT", "ETH/USDT"],
                    execution_mode="threadpool",
                )

    assert isinstance(long_df, pd.DataFrame)
    assert isinstance(short_df, pd.DataFrame)
