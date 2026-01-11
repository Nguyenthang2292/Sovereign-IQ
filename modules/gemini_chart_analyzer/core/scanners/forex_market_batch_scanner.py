
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from modules.common.core.forex_data_fetcher import ForexDataFetcher
from modules.common.ui.logging import log_error, log_warn
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner
from modules.common.ui.logging import log_error, log_warn
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner

"""
Forex Market Batch Scanner using TradingView scraper.

Extends MarketBatchScanner to use TradingView scraper for forex data
with format: OANDA:<SYMBOL_NAME>
"""





class DataFetchStrategy(ABC):
    """Abstract base class for data fetch strategies."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        check_freshness: bool = False,
        exchanges: Optional[List[str]] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT', 'EUR/USD')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch
            check_freshness: Whether to check data freshness
            exchanges: Optional list of exchanges to try

        Returns:
            Tuple[pd.DataFrame, str]: DataFrame with OHLCV data and source string.
            Returns (None, None) if data cannot be fetched.
        """
        pass


class ForexDataFetchStrategy(DataFetchStrategy):
    """Strategy for fetching forex data using TradingView."""

    def __init__(self):
        """Initialize forex data fetcher."""
        self.forex_data_fetcher = ForexDataFetcher()
        if not self.forex_data_fetcher.is_available():
            log_warn("TradingView scraper is not available. Install it with: pip install tradingview-scraper")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        check_freshness: bool = False,
        exchanges: Optional[List[str]] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch OHLCV data for a forex symbol using TradingView.

        Args:
            symbol: Forex symbol (e.g., 'EUR/USD', 'GBP/USD')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            limit: Number of candles
            check_freshness: Not used for forex (kept for compatibility)
            exchanges: Not used for forex (kept for compatibility)

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: DataFrame and source, or (None, None) if failed
        """
        if not self.forex_data_fetcher.is_available():
            log_error(f"TradingView scraper not available for {symbol}")
            return None, None

        try:
            df, source = self.forex_data_fetcher.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
            return df, source
        except Exception as e:
            log_error(f"Error fetching {symbol} from TradingView: {e}")
            return None, None


class CryptoDataFetchStrategy(DataFetchStrategy):
    """Strategy for fetching crypto data using ccxt exchanges."""

    def __init__(self, data_fetcher):
        """
        Initialize crypto data fetch strategy.

        Args:
            data_fetcher: DataFetcher instance for crypto data
        """
        self.data_fetcher = data_fetcher

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        check_freshness: bool = False,
        exchanges: Optional[List[str]] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch OHLCV data for a crypto symbol using ccxt exchanges.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            limit: Number of candles
            check_freshness: Whether to check data freshness
            exchanges: Optional list of exchanges to try

        Returns:
            Tuple[pd.DataFrame, str]: DataFrame with OHLCV data and source string.
            Returns (None, None) if data cannot be fetched.
        """
        return self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol, limit, timeframe, check_freshness, exchanges
        )


class ForexMarketBatchScanner(MarketBatchScanner):
    """Market Batch Scanner for Forex using TradingView scraper."""

    def __init__(
        self,
        charts_per_batch: int = 100,
        cooldown_seconds: float = 2.5,
        min_candles: Optional[int] = None,
        fetch_strategy: Optional[DataFetchStrategy] = None,
    ):
        """
        Initialize ForexMarketBatchScanner.

        Args:
            charts_per_batch: Number of charts per batch (default: 100)
            cooldown_seconds: Cooldown between batch requests (default: 2.5s)
            min_candles: Minimum number of candles required (default: 20)
            fetch_strategy: Data fetch strategy (default: ForexDataFetchStrategy)
        """
        # Initialize parent with dummy exchange (not used for forex)
        super().__init__(
            charts_per_batch=charts_per_batch,
            cooldown_seconds=cooldown_seconds,
            quote_currency="USD",
            exchange_name="oanda",
            min_candles=min_candles,
        )

        # Use strategy pattern for data fetching
        self.fetch_strategy = fetch_strategy or ForexDataFetchStrategy()

    def _fetch_forex_data(self, symbol: str, timeframe: str, limit: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch OHLCV data for a single forex symbol using configured strategy.

        Args:
            symbol: Forex symbol (e.g., 'EUR/USD', 'GBP/USD')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            limit: Number of candles

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: DataFrame and source, or (None, None) if failed
        """
        return self.fetch_strategy.fetch_ohlcv(symbol, timeframe, limit)

    def _fetch_batch_data(self, symbols: List[str], timeframe: str, limit: int) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV data for a batch of forex symbols using configured strategy.

        Args:
            symbols: List of forex symbols to fetch (e.g., ['EUR/USD', 'GBP/USD'])
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            limit: Number of candles

        Returns:
            List of dicts with 'symbol' and 'df' keys
        """
        symbols_data = []

        for symbol in symbols:
            df, source = self._fetch_forex_data(symbol, timeframe, limit)

            if df is not None and not df.empty and len(df) >= self.min_candles:
                symbols_data.append({"symbol": symbol, "df": df})
            else:
                log_warn(f"Insufficient data for {symbol}, skipping...")

        return symbols_data

    def set_fetch_strategy(self, strategy: DataFetchStrategy):
        """
        Set the data fetch strategy.

        Args:
            strategy: Data fetch strategy instance
        """
        self.fetch_strategy = strategy

