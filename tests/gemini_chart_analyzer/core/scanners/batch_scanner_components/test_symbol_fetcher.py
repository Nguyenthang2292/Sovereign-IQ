"""
Tests for SymbolFetcher (batch scanner component).

Covers symbol fetching, retry logic, and timeout/RequestTimeout as retryable.
"""

from unittest.mock import Mock

import pytest

try:
    import ccxt

    RequestTimeout = ccxt.RequestTimeout
except ImportError:
    RequestTimeout = None

from modules.gemini_chart_analyzer.core.exceptions import DataFetchError
from modules.gemini_chart_analyzer.core.scanners.batch_scanner_components.symbol_fetcher import SymbolFetcher


@pytest.fixture
def sample_markets():
    """Sample markets dict for spot USDT symbols."""
    return {
        "BTC/USDT": {"quote": "USDT", "active": True, "type": "spot"},
        "ETH/USDT": {"quote": "USDT", "active": True, "type": "spot"},
    }


def test_get_all_symbols_success(sample_markets):
    """SymbolFetcher returns spot USDT symbols when load_markets succeeds."""
    fetcher = SymbolFetcher(exchange_name="binance", quote_currency="USDT")
    mock_exchange = Mock()
    mock_exchange.load_markets = Mock(return_value=sample_markets)
    mock_exchange.timeout = 10000
    fetcher.public_exchange_manager.connect_to_exchange_with_no_credentials = Mock(return_value=mock_exchange)

    symbols = fetcher.get_all_symbols(max_retries=2)

    assert symbols == ["BTC/USDT", "ETH/USDT"]
    mock_exchange.load_markets.assert_called_once()


def test_get_all_symbols_request_timeout_retry_then_success(sample_markets):
    """RequestTimeout (e.g. from dapi.binance.com) is retryable; succeeds on second attempt."""
    if RequestTimeout is None:
        pytest.skip("ccxt not available")
    fetcher = SymbolFetcher(exchange_name="binance", quote_currency="USDT")
    mock_exchange = Mock()
    mock_exchange.load_markets = Mock(
        side_effect=[
            RequestTimeout("binance GET https://dapi.binance.com/dapi/v1/exchangeInfo"),
            sample_markets,
        ]
    )
    mock_exchange.timeout = 10000
    fetcher.public_exchange_manager.connect_to_exchange_with_no_credentials = Mock(return_value=mock_exchange)

    symbols = fetcher.get_all_symbols(max_retries=3, retry_delay=0.01)

    assert symbols == ["BTC/USDT", "ETH/USDT"]
    assert mock_exchange.load_markets.call_count == 2


def test_get_all_symbols_request_timeout_exhausted_raises(sample_markets):
    """When all retries fail with RequestTimeout, DataFetchError is raised."""
    if RequestTimeout is None:
        pytest.skip("ccxt not available")
    fetcher = SymbolFetcher(exchange_name="binance", quote_currency="USDT")
    mock_exchange = Mock()
    mock_exchange.load_markets = Mock(
        side_effect=RequestTimeout("binance GET https://dapi.binance.com/dapi/v1/exchangeInfo")
    )
    mock_exchange.timeout = 10000
    fetcher.public_exchange_manager.connect_to_exchange_with_no_credentials = Mock(return_value=mock_exchange)

    with pytest.raises(DataFetchError, match="Failed to fetch symbols after 2 attempts"):
        fetcher.get_all_symbols(max_retries=2, retry_delay=0.01)

    assert mock_exchange.load_markets.call_count == 2


def test_get_all_symbols_filters_spot_only(sample_markets):
    """Only spot markets with quote USDT are returned."""
    markets = {
        **sample_markets,
        "BTC/USD": {"quote": "USD", "active": True, "type": "spot"},
        "ETH/BUSD": {"quote": "BUSD", "active": True, "type": "spot"},
        "XRP/USDT": {"quote": "USDT", "active": False, "type": "spot"},
        "BTC/USDT:USDT": {"quote": "USDT", "active": True, "type": "future"},
    }
    fetcher = SymbolFetcher(exchange_name="binance", quote_currency="USDT")
    mock_exchange = Mock()
    mock_exchange.load_markets = Mock(return_value=markets)
    mock_exchange.timeout = 10000
    fetcher.public_exchange_manager.connect_to_exchange_with_no_credentials = Mock(return_value=mock_exchange)

    symbols = fetcher.get_all_symbols()

    assert symbols == ["BTC/USDT", "ETH/USDT"]
    assert "BTC/USD" not in symbols
    assert "ETH/BUSD" not in symbols
    assert "XRP/USDT" not in symbols
    assert "BTC/USDT:USDT" not in symbols
