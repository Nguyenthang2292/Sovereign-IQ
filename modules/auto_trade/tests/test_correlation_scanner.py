"""
Tests for CorrelationScanner.

Unit tests for correlation scanning, hedge ratio calculation,
and cache management.
"""

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.auto_trade.execution.correlation_scanner import (
    CorrelationScanner,
    HedgeCandidate,
)


def test_correlation_scanner_initialization():
    """Test CorrelationScanner initialization with default values."""
    scanner = CorrelationScanner()

    assert scanner.min_correlation == CorrelationScanner.DEFAULT_MIN_CORRELATION
    assert scanner.lookback == CorrelationScanner.DEFAULT_LOOKBACK
    assert scanner.timeframe == CorrelationScanner.DEFAULT_TIMEFRAME
    assert scanner.refresh_interval == CorrelationScanner.DEFAULT_REFRESH_INTERVAL


def test_correlation_scanner_custom_settings():
    """Test CorrelationScanner with custom settings."""
    scanner = CorrelationScanner(
        min_correlation=0.80,
        lookback=200,
        timeframe="4h",
        refresh_interval=3600,
    )

    assert scanner.min_correlation == 0.80
    assert scanner.lookback == 200
    assert scanner.timeframe == "4h"
    assert scanner.refresh_interval == 3600


def test_make_cache_key():
    """Test cache key generation."""
    scanner = CorrelationScanner()

    key = scanner._make_cache_key("BTC/USDT", "ETH/USDT")
    assert key == "BTC/USDT:ETH/USDT"

    key_reversed = scanner._make_cache_key("ETH/USDT", "BTC/USDT")
    assert key_reversed == "ETH/USDT:BTC/USDT"


def test_is_cache_valid():
    """Test cache validity check."""
    scanner = CorrelationScanner()

    assert not scanner._is_cache_valid("BTC/USDT", "ETH/USDT")

    current_time = time.time()
    scanner._cache["BTC/USDT:ETH/USDT"] = {
        "correlation": 0.75,
        "cached_at": current_time,
    }

    scanner.refresh_interval = 999999999

    assert scanner._is_cache_valid("BTC/USDT", "ETH/USDT")

    scanner.refresh_interval = 1
    scanner._cache["BTC/USDT:ETH/USDT"]["cached_at"] = current_time - 10
    assert not scanner._is_cache_valid("BTC/USDT", "ETH/USDT")


def test_should_refresh_full_cache():
    """Test full cache refresh check."""
    scanner = CorrelationScanner()

    assert scanner._should_refresh_full_cache()

    scanner._last_cache_refresh = 99999999999
    scanner.refresh_interval = 1
    assert not scanner._should_refresh_full_cache()


def test_get_default_symbol_pool():
    """Test default symbol pool generation."""
    scanner = CorrelationScanner()

    pool = scanner._get_default_symbol_pool("BTC/USDT")

    assert "BTC/USDT" not in pool
    assert "ETH/USDT" in pool
    assert len(pool) > 0


def test_cache_stats():
    """Test cache statistics."""
    scanner = CorrelationScanner()

    stats = scanner.get_cache_stats()

    assert "cache_size" in stats
    assert "last_refresh" in stats
    assert "refresh_interval" in stats
    assert stats["refresh_interval"] == CorrelationScanner.DEFAULT_REFRESH_INTERVAL


def test_refresh_correlation_cache_all():
    """Test refreshing all cache."""
    scanner = CorrelationScanner()

    scanner._cache["BTC/USDT:ETH/USDT"] = {"correlation": 0.75}
    scanner._last_cache_refresh = 1234567890

    scanner.refresh_correlation_cache()

    assert len(scanner._cache) == 0
    assert scanner._last_cache_refresh is None


def test_refresh_correlation_cache_specific():
    """Test refreshing specific symbols from cache."""
    scanner = CorrelationScanner()

    scanner._cache = {
        "BTC/USDT:ETH/USDT": {"correlation": 0.75},
        "BTC/USDT:BNB/USDT": {"correlation": 0.65},
        "ETH/USDT:SOL/USDT": {"correlation": 0.55},
    }

    scanner.refresh_correlation_cache(["BTC/USDT"])

    assert "BTC/USDT:ETH/USDT" not in scanner._cache
    assert "BTC/USDT:BNB/USDT" not in scanner._cache
    assert "ETH/USDT:SOL/USDT" in scanner._cache


def test_hedge_candidate_dataclass():
    """Test HedgeCandidate dataclass."""
    candidate = HedgeCandidate(
        symbol="ETH/USDT",
        correlation=0.75,
        hedge_ratio=1.2,
        kalman_hedge_ratio=1.15,
        score=0.9,
        regime="STAT_ARB",
    )

    assert candidate.symbol == "ETH/USDT"
    assert candidate.correlation == 0.75
    assert candidate.hedge_ratio == 1.2
    assert candidate.kalman_hedge_ratio == 1.15
    assert candidate.score == 0.9
    assert candidate.regime == "STAT_ARB"


@patch("modules.auto_trade.execution.correlation_scanner.DataFetcher")
def test_scan_hedge_candidates_no_data(mock_data_fetcher):
    """Test scanning with no data fetcher returns empty list."""
    mock_fetcher = MagicMock()
    mock_fetcher.fetch_ohlcv.return_value = None

    scanner = CorrelationScanner(data_fetcher=mock_fetcher, min_correlation=0.50)

    candidates = scanner.scan_hedge_candidates("BTC/USDT", ["ETH/USDT"])

    assert candidates == []


def test_correlation_min_correlation_bounds():
    """Test min_correlation bounds (0.50-0.90)."""
    with pytest.raises(ValueError):
        CorrelationScanner(min_correlation=0.30)

    with pytest.raises(ValueError):
        CorrelationScanner(min_correlation=1.0)

    scanner = CorrelationScanner(min_correlation=0.50)
    assert scanner.min_correlation == 0.50

    scanner = CorrelationScanner(min_correlation=0.90)
    assert scanner.min_correlation == 0.90


def test_calculate_hedge_ratio_uses_cached_value_without_refetch():
    """Second hedge ratio call should hit cache and avoid extra fetches."""
    mock_fetcher = MagicMock()
    prices = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
    mock_fetcher.fetch_ohlcv.return_value = prices

    scanner = CorrelationScanner(data_fetcher=mock_fetcher, refresh_interval=999999)

    with patch("modules.auto_trade.execution.correlation_scanner.calculate_ols_hedge_ratio", return_value=1.25):
        first = scanner.calculate_hedge_ratio("BTC/USDT", "ETH/USDT", method="OLS")
        second = scanner.calculate_hedge_ratio("BTC/USDT", "ETH/USDT", method="OLS")

    assert first == 1.25
    assert second == 1.25
    assert mock_fetcher.fetch_ohlcv.call_count == 2


def test_scan_hedge_candidates_normalizes_perp_symbols():
    """PERP symbols should normalize to CCXT pair format."""
    scanner = CorrelationScanner(min_correlation=0.50)
    scanner.calculate_correlation = MagicMock(return_value=0.80)  # type: ignore[method-assign]
    scanner.calculate_hedge_ratio = MagicMock(side_effect=[1.1, 1.0])  # type: ignore[method-assign]

    candidates = scanner.scan_hedge_candidates("BTC-PERP", ["ETH-PERP"])

    assert len(candidates) == 1
    assert candidates[0].symbol == "ETH/USDT"
    scanner.calculate_correlation.assert_called_once_with("BTC/USDT", "ETH/USDT")
