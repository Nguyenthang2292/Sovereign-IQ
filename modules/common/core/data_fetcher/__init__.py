"""
DataFetcher module - refactored with composition pattern for better modularity.

This module provides the DataFetcher class, which is responsible for fetching and caching market data
(prices, OHLCV) from cryptocurrency exchanges via the ExchangeManager interface. It handles efficient
price lookups, manages shutdown signals for clean operation, and supports real-time data acquisition
such as current prices and historical candles.

Architecture:
    The DataFetcher class uses composition to delegate specialized functionality to focused components:
    - BinancePriceFetcher: Current price fetching from Binance
    - BinanceFuturesFetcher: Futures positions and balance operations
    - SymbolDiscovery: Symbol discovery for spot and futures markets
    - OHLCVFetcher: OHLCV data fetching with exchange fallback

    This design provides better separation of concerns while maintaining exact backward compatibility
    with the original monolithic implementation.
"""

from typing import Dict, List, Optional

from modules.common.core.exchange_manager import ExchangeManager

from .base import DataFetcherBase
from .binance_futures import BinanceFuturesFetcher
from .binance_prices import BinancePriceFetcher
from .exceptions import SymbolFetchError
from .ohlcv import OHLCVFetcher
from .symbol_discovery import SymbolDiscovery

__all__ = ["DataFetcher", "SymbolFetchError"]


class DataFetcher(DataFetcherBase):
    """
    Fetches market data (prices, OHLCV) from exchanges.

    This class provides a unified interface for all data fetching operations, delegating
    to specialized components for different types of operations while maintaining the
    original public API for backward compatibility.
    """

    def __init__(self, exchange_manager: ExchangeManager, shutdown_event=None):
        """
        Initialize DataFetcher with specialized components.

        Args:
            exchange_manager: ExchangeManager instance for exchange operations
            shutdown_event: Optional threading.Event for graceful shutdown
        """
        super().__init__(exchange_manager, shutdown_event)

        # Initialize specialized fetchers
        self._binance_prices = BinancePriceFetcher(self)
        self._binance_futures = BinanceFuturesFetcher(self)
        self._symbol_discovery = SymbolDiscovery(self)
        self._ohlcv = OHLCVFetcher(self)

    # ========== Binance Prices Methods ==========

    def fetch_current_prices_from_binance(self, symbols: list):
        """
        Fetches current prices for all symbols from Binance.

        Args:
            symbols: List of trading symbols to fetch prices for

        Returns:
            None. Updates self.market_prices dict with fetched prices.
        """
        return self._binance_prices.fetch_current_prices_from_binance(symbols)

    # ========== Binance Futures Methods ==========

    def fetch_binance_futures_positions(
        self,
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = False,
        debug: bool = False,
    ) -> List[Dict]:
        """
        Fetches open positions from Binance Futures USDT-M.

        Args:
            api_key: API Key from Binance
            api_secret: API Secret from Binance
            testnet: Use testnet if True (default: False)
            debug: Show debug info if True (default: False)

        Returns:
            List of dictionaries containing position information
        """
        return self._binance_futures.fetch_binance_futures_positions(api_key, api_secret, testnet, debug)

    def fetch_binance_account_balance(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        currency: str = "USDT",
    ) -> Optional[float]:
        """
        Fetch account balance from Binance Futures USDT-M.

        Args:
            api_key: API Key from Binance (optional)
            api_secret: API Secret from Binance (optional)
            testnet: Use testnet if True (default: False)
            currency: Currency to fetch balance for (default: "USDT")

        Returns:
            Account balance as float, or None if error
        """
        return self._binance_futures.fetch_binance_account_balance(api_key, api_secret, testnet, currency)

    # ========== Symbol Discovery Methods ==========

    def list_binance_futures_symbols(
        self,
        exclude_symbols: Optional[set] = None,
        max_candidates: Optional[int] = None,
        progress_label: str = "Symbol Discovery",
    ) -> List[str]:
        """
        Lists available Binance USDT-M futures symbols sorted by quote volume.

        Args:
            exclude_symbols: Symbols to exclude
            max_candidates: Optional maximum number of symbols to return
            progress_label: Label for the progress bar

        Returns:
            List of symbol strings sorted by descending volume
        """
        return self._symbol_discovery.list_binance_futures_symbols(exclude_symbols, max_candidates, progress_label)

    def get_spot_symbols(
        self,
        exchange_name: str = "binance",
        quote_currency: str = "USDT",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> List[str]:
        """
        Get all spot trading symbols from exchange with retry logic.

        Args:
            exchange_name: Exchange name (default: 'binance')
            quote_currency: Quote currency to filter (default: 'USDT')
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Base delay between retries in seconds (default: 1.0)

        Returns:
            Sorted list of spot trading symbols

        Raises:
            SymbolFetchError: If fetching symbols fails after all retries
        """
        return self._symbol_discovery.get_spot_symbols(exchange_name, quote_currency, max_retries, retry_delay)

    # ========== OHLCV Methods ==========

    @staticmethod
    def dataframe_to_close_series(df):
        """
        Converts a fetched OHLCV DataFrame into a pandas Series of closing prices.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series of closing prices indexed by timestamp
        """
        return OHLCVFetcher.dataframe_to_close_series(df)

    def fetch_ohlcv_with_fallback_exchange(
        self,
        symbol,
        limit=1500,
        timeframe="1h",
        check_freshness=False,
        exchanges=None,
    ):
        """
        Fetches OHLCV data using ccxt with fallback exchanges (with caching).

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Number of candles to fetch (default: 1500)
            timeframe: Timeframe string (e.g., '1h', '1d')
            check_freshness: If True, checks data freshness (default: False)
            exchanges: Optional list of exchange IDs to try

        Returns:
            Tuple[pd.DataFrame, str]: DataFrame with OHLCV data and exchange_id,
            or (None, None) if data cannot be fetched
        """
        return self._ohlcv.fetch_ohlcv_with_fallback_exchange(symbol, limit, timeframe, check_freshness, exchanges)
