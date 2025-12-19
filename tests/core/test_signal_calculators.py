"""
Test file for core.signal_calculators module.

This test file focuses on testing get_range_oscillator_signal with mock data
to avoid loading full market data which is very slow.

Run with: python -m pytest tests/core/test_signal_calculators.py -v
Or: python tests/core/test_signal_calculators.py
"""

import sys
import warnings
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add project root to path (same as test_main_voting.py)
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

# Import after path setup
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from core.signal_calculators import (
    get_range_oscillator_signal,
    get_spc_signal,
    get_xgboost_signal,
    get_hmm_signal,
)


def create_mock_ohlcv_data(limit: int = 100) -> pd.DataFrame:
    """Create mock OHLCV data for testing."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq='1h')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 50000.0
    prices = []
    for i in range(limit):
        change = np.random.randn() * 100
        base_price = max(100, base_price + change)
        high = base_price * (1 + abs(np.random.randn() * 0.01))
        low = base_price * (1 - abs(np.random.randn() * 0.01))
        close = base_price + np.random.randn() * 50
        volume = np.random.uniform(1000, 10000)
        prices.append({
            'timestamp': dates[i],
            'open': base_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(prices)
    df.set_index('timestamp', inplace=True)
    return df


def create_mock_data_fetcher(df: pd.DataFrame) -> DataFetcher:
    """Create a mock DataFetcher that returns the provided DataFrame."""
    exchange_manager = Mock(spec=ExchangeManager)
    data_fetcher = DataFetcher(exchange_manager)
    
    # Mock the fetch method to return our test data
    data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(
        return_value=(df, "binance")
    )
    
    return data_fetcher


def test_get_range_oscillator_signal_success():
    """Test successful range oscillator signal calculation."""
    print("\n=== Test: get_range_oscillator_signal - Success ===")
    
    df = create_mock_ohlcv_data(limit=100)
    data_fetcher = create_mock_data_fetcher(df)
    
    result = get_range_oscillator_signal(
        data_fetcher=data_fetcher,
        symbol="BTC/USDT",
        timeframe="1h",
        limit=100,
        osc_length=50,
        osc_mult=2.0,
    )
    
    print(f"Result: {result}")
    assert result is not None, "Signal should not be None"
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 2, "Result should have 2 elements (signal, confidence)"
    assert isinstance(result[0], int), "Signal should be an integer"
    assert isinstance(result[1], float), "Confidence should be a float"
    assert result[0] in [-1, 0, 1], "Signal should be -1, 0, or 1"
    assert 0.0 <= result[1] <= 1.0, "Confidence should be between 0 and 1"
    
    print("[OK] Test passed: Signal calculated successfully")


def test_get_range_oscillator_signal_empty_dataframe():
    """Test range oscillator signal with empty DataFrame."""
    print("\n=== Test: get_range_oscillator_signal - Empty DataFrame ===")
    
    empty_df = pd.DataFrame()
    data_fetcher = create_mock_data_fetcher(empty_df)
    
    result = get_range_oscillator_signal(
        data_fetcher=data_fetcher,
        symbol="BTC/USDT",
        timeframe="1h",
        limit=100,
    )
    
    print(f"Result: {result}")
    assert result is None, "Signal should be None for empty DataFrame"
    print("[OK] Test passed: Empty DataFrame handled correctly")


def test_get_range_oscillator_signal_missing_columns():
    """Test range oscillator signal with missing required columns."""
    print("\n=== Test: get_range_oscillator_signal - Missing Columns ===")
    
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'volume': [1000, 1100, 1200]
        # Missing 'high', 'low', 'close'
    })
    data_fetcher = create_mock_data_fetcher(df)
    
    result = get_range_oscillator_signal(
        data_fetcher=data_fetcher,
        symbol="BTC/USDT",
        timeframe="1h",
        limit=100,
    )
    
    print(f"Result: {result}")
    assert result is None, "Signal should be None for missing columns"
    print("[OK] Test passed: Missing columns handled correctly")


def test_get_range_oscillator_signal_none_dataframe():
    """Test range oscillator signal with None DataFrame."""
    print("\n=== Test: get_range_oscillator_signal - None DataFrame ===")
    
    data_fetcher = create_mock_data_fetcher(None)
    
    result = get_range_oscillator_signal(
        data_fetcher=data_fetcher,
        symbol="BTC/USDT",
        timeframe="1h",
        limit=100,
    )
    
    print(f"Result: {result}")
    assert result is None, "Signal should be None for None DataFrame"
    print("[OK] Test passed: None DataFrame handled correctly")


def test_get_range_oscillator_signal_exception_handling():
    """Test range oscillator signal exception handling."""
    print("\n=== Test: get_range_oscillator_signal - Exception Handling ===")
    
    df = create_mock_ohlcv_data(limit=100)
    data_fetcher = create_mock_data_fetcher(df)
    
    # Mock to raise an exception
    data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(
        side_effect=Exception("Test exception")
    )
    
    result = get_range_oscillator_signal(
        data_fetcher=data_fetcher,
        symbol="BTC/USDT",
        timeframe="1h",
        limit=100,
    )
    
    print(f"Result: {result}")
    assert result is None, "Signal should be None when exception occurs"
    print("[OK] Test passed: Exception handled correctly")


def test_get_range_oscillator_signal_with_strategies():
    """Test range oscillator signal with specific strategies."""
    print("\n=== Test: get_range_oscillator_signal - With Strategies ===")
    
    df = create_mock_ohlcv_data(limit=100)
    data_fetcher = create_mock_data_fetcher(df)
    
    result = get_range_oscillator_signal(
        data_fetcher=data_fetcher,
        symbol="BTC/USDT",
        timeframe="1h",
        limit=100,
        strategies=[2, 3, 4],
    )
    
    print(f"Result: {result}")
    assert result is None or isinstance(result, tuple), "Result should be None or tuple"
    if result is not None:
        assert len(result) == 2, "Result should have 2 elements"
    print("[OK] Test passed: Strategies parameter handled correctly")


def test_get_range_oscillator_signal_insufficient_data():
    """Test range oscillator signal with insufficient data."""
    print("\n=== Test: get_range_oscillator_signal - Insufficient Data ===")
    
    # Create data with only 10 rows (less than osc_length=50)
    df = create_mock_ohlcv_data(limit=10)
    data_fetcher = create_mock_data_fetcher(df)
    
    result = get_range_oscillator_signal(
        data_fetcher=data_fetcher,
        symbol="BTC/USDT",
        timeframe="1h",
        limit=10,
        osc_length=50,  # More than available data
    )
    
    print(f"Result: {result}")
    # Result might be None or might still work with limited data
    assert result is None or isinstance(result, tuple), "Result should be None or tuple"
    print("[OK] Test passed: Insufficient data handled correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Testing core.signal_calculators.get_range_oscillator_signal")
    print("=" * 80)
    
    tests = [
        test_get_range_oscillator_signal_success,
        test_get_range_oscillator_signal_empty_dataframe,
        test_get_range_oscillator_signal_missing_columns,
        test_get_range_oscillator_signal_none_dataframe,
        test_get_range_oscillator_signal_exception_handling,
        test_get_range_oscillator_signal_with_strategies,
        test_get_range_oscillator_signal_insufficient_data,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] Test error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

