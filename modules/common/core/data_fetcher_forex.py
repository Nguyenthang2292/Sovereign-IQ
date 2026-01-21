"""
Forex Data Fetcher using TradingView scraper with tvDatafeed fallback.

Fetches forex OHLCV data using:
1. tradingview_scraper library with format: "EXCHANGE:<SYMBOL_NAME>" (OANDA, FXCM, FOREXCOM, IC Markets, Pepperstone)
2. tvDatafeed library as automatic fallback if tradingview_scraper fails

Supports multiple forex exchanges:
- OANDA (default)
- FXCM
- FOREXCOM (Forex.com)
- IC Markets
- Pepperstone
"""

import asyncio
import os
import threading
import time
from typing import Optional, Tuple

import pandas as pd

from modules.common.ui.logging import (
    log_error,
    log_info,
    log_success,
    log_warn,
)


class ForexDataFetcher:
    """Fetches forex market data using TradingView scraper with tvDatafeed fallback."""

    # Supported forex exchanges (priority order - tried first to last)
    SUPPORTED_FOREX_EXCHANGES = [
        "OANDA",
        "FXCM",
        "FOREXCOM",
        "IC MARKETS",
        "PEPPERSTONE",
    ]

    def __init__(
        self,
        tradingview_username: Optional[str] = None,
        tradingview_password: Optional[str] = None,
        request_pause: float = 1.0,
    ):
        """
        Initialize ForexDataFetcher with both data sources.

        Args:
            tradingview_username: TradingView username for tvDatafeed login (optional).
                                  Uses TRADINGVIEW_USERNAME env var if not provided.
            tradingview_password: TradingView password for tvDatafeed login (optional).
                                  Uses TRADINGVIEW_PASSWORD env var if not provided.
            request_pause: Minimum delay between API calls in seconds (default: 1.0).
        """
        # Get credentials from parameters or environment variables
        username = tradingview_username or os.getenv("TRADINGVIEW_USERNAME")
        password = tradingview_password or os.getenv("TRADINGVIEW_PASSWORD")

        # Rate limiting
        self.request_pause = request_pause
        self._request_lock = threading.Lock()
        self._last_request_ts = 0.0

        # Try tvDatafeed first (primary source)
        self.tv_datafeed = None
        self._tvdatafeed_available = False
        try:
            from tvDatafeed import Interval, TvDatafeed

            # Initialize with credentials if available, otherwise use anonymous mode
            if username and password:
                try:
                    self.tv_datafeed = TvDatafeed(username, password)
                    log_info("tvDatafeed initialized successfully with login credentials (primary source)")
                except Exception as e:
                    log_warn(f"Failed to login to tvDatafeed with credentials: {e}, trying anonymous mode...")
                    self.tv_datafeed = TvDatafeed()
                    log_info("tvDatafeed initialized in anonymous mode (primary source)")
            else:
                self.tv_datafeed = TvDatafeed()
                log_info("tvDatafeed initialized in anonymous mode (primary source)")
                if not username or not password:
                    log_info(
                        "Tip: Set TRADINGVIEW_USERNAME and TRADINGVIEW_PASSWORD "
                        "environment variables for authenticated access"
                    )

            self.Interval = Interval
            self._tvdatafeed_available = True
        except ImportError:
            log_warn("tvDatafeed not available. Will use tradingview_scraper fallback.")

        # Try tradingview_scraper as fallback
        self.real_time_data = None
        self._tradingview_scraper_available = False
        try:
            from tradingview_scraper.symbols.stream import RealTimeData

            self.real_time_data = RealTimeData()
            self._tradingview_scraper_available = True
            log_info("tradingview_scraper initialized successfully (fallback)")
        except ImportError:
            log_warn("tradingview_scraper not available. Install it with: pip install tradingview-scraper")

        if not self._tvdatafeed_available and not self._tradingview_scraper_available:
            log_error("Neither tvDatafeed nor tradingview_scraper is available!")

    def _throttled_call(self, func, *args, **kwargs):
        """
        Ensures a minimum delay between REST calls to respect rate limits.

        Args:
            func: Function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func(*args, **kwargs)
        """
        with self._request_lock:
            wait = self.request_pause - (time.time() - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            result = func(*args, **kwargs)
            self._last_request_ts = time.time()
            return result

    def is_available(self) -> bool:
        """Check if at least one data source is available."""
        return self._tradingview_scraper_available or self._tvdatafeed_available

    def _convert_forex_symbol_to_tradingview(self, symbol: str, exchange: str = "OANDA") -> str:
        """
        Convert forex symbol to TradingView format.

        Args:
            symbol: Forex symbol (e.g., 'EUR/USD', 'GBP/USD')
            exchange: Exchange name (default: 'OANDA')

        Returns:
            TradingView exchange_symbol format (e.g., 'OANDA:EURUSD')
        """
        # Remove '/' from symbol (EUR/USD -> EURUSD)
        clean_symbol = symbol.replace("/", "").replace("-", "")
        return f"{exchange}:{clean_symbol}"

    def _convert_timeframe(self, timeframe: str, target_format: str = "tradingview"):
        """
        Convert timeframe to target format.

        Args:
            timeframe: Timeframe string (e.g., '1h', '4h', '1d', '15m')
            target_format: Target format ('tradingview' or 'tvdatafeed')

        Returns:
            TradingView timeframe string or tvDatafeed Interval enum
        """
        timeframe_lower = timeframe.lower()

        if target_format == "tvdatafeed":
            if not self._tvdatafeed_available:
                return None
            try:
                from tvDatafeed import Interval
            except ImportError:
                return None
        else:
            Interval = None

        # Minutes
        if timeframe_lower.endswith("m"):
            minutes = int(timeframe_lower[:-1])
            if target_format == "tradingview":
                mapping = {1: "1", 5: "5", 15: "15", 30: "30", 60: "60"}
            else:
                mapping = {
                    1: Interval.in_1_minute,
                    5: Interval.in_5_minute,
                    15: Interval.in_15_minute,
                    30: Interval.in_30_minute,
                }
            result = mapping.get(minutes)
            if result is None:
                log_warn(f"Unknown minute timeframe {timeframe}, using 15m")
                return mapping[15] if target_format == "tradingview" else Interval.in_15_minute
            return result

        # Hours
        elif timeframe_lower.endswith("h"):
            hours = int(timeframe_lower[:-1])
            if target_format == "tradingview":
                mapping = {1: "60", 4: "240"}
            else:
                mapping = {1: Interval.in_1_hour, 4: Interval.in_4_hour}
            result = mapping.get(hours)
            if result is None:
                log_warn(f"Unknown hour timeframe {timeframe}, using 1h")
                return mapping[1] if target_format == "tradingview" else Interval.in_1_hour
            return result

        # Days
        elif timeframe_lower.endswith("d"):
            days = int(timeframe_lower[:-1])
            if days == 1:
                return "D" if target_format == "tradingview" else Interval.in_daily
            log_warn(f"Unknown day timeframe {timeframe}, using 1d")
            return "D" if target_format == "tradingview" else Interval.in_daily

        # Weeks
        elif timeframe_lower.endswith("w"):
            return "W" if target_format == "tradingview" else Interval.in_weekly

        # Default fallback
        log_warn(f"Unknown timeframe format {timeframe}, using 1h")
        return "60" if target_format == "tradingview" else Interval.in_1_hour

    def _convert_timeframe_to_tradingview(self, timeframe: str) -> str:
        """
        Convert timeframe to TradingView format.

        Args:
            timeframe: Timeframe string (e.g., '1h', '4h', '1d', '15m')

        Returns:
            TradingView timeframe format
        """
        return self._convert_timeframe(timeframe, "tradingview")

    def _convert_timeframe_to_tvdatafeed(self, timeframe: str):
        """
        Convert timeframe to tvDatafeed Interval enum.

        Args:
            timeframe: Timeframe string (e.g., '1h', '4h', '1d', '15m')

        Returns:
            tvDatafeed Interval enum value
        """
        return self._convert_timeframe(timeframe, "tvdatafeed")

    def _convert_forex_symbol_for_tvdatafeed(self, symbol: str, exchange: str = "OANDA") -> Tuple[str, str]:
        """
        Convert forex symbol for tvDatafeed.

        Args:
            symbol: Forex symbol (e.g., 'EUR/USD', 'GBP/USD')
            exchange: Exchange name (default: 'OANDA')

        Returns:
            Tuple[symbol, exchange]: Symbol and exchange for tvDatafeed
        """
        # Remove '/' from symbol (EUR/USD -> EURUSD)
        clean_symbol = symbol.replace("/", "").replace("-", "")
        # Normalize exchange name for tvDatafeed
        # Some exchanges might need normalization (e.g., "IC MARKETS" -> "ICMARKETS")
        normalized_exchange = exchange.replace(" ", "").upper()
        return clean_symbol, normalized_exchange

    def _process_tvdatafeed_result(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        limit: int,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Process tvDatafeed result DataFrame to standard format.

        Args:
            df: Raw DataFrame from tvDatafeed
            symbol: Original symbol (e.g., 'EUR/USD')
            exchange: Exchange name used
            limit: Requested limit

        Returns:
            Tuple[pd.DataFrame, str]: Processed DataFrame and source string
        """
        try:
            # tvDatafeed returns DataFrame with datetime index and OHLCV columns
            # Ensure column names match our expected format
            df_columns_lower = {col.lower(): col for col in df.columns}

            # Map columns to standard format
            column_mapping = {}
            for expected_col in ["open", "high", "low", "close", "volume"]:
                if expected_col in df_columns_lower:
                    column_mapping[df_columns_lower[expected_col]] = expected_col

            # Rename columns to standard format
            if column_mapping:
                df = df.rename(columns=column_mapping)

            # Ensure we have required columns
            required_cols = ["open", "high", "low", "close"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                log_error(f"Missing required columns in tvDatafeed data: {missing_cols}")
                return None, None

            # Add volume column if missing (set to 0)
            if "volume" not in df.columns:
                df["volume"] = 0
                log_warn(f"Volume data not available for {symbol}, setting to 0")

            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            else:
                df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index

            # Sort by index
            df.sort_index(inplace=True)

            # Limit to requested number of bars
            if len(df) > limit:
                df = df.tail(limit)

            source = f"tvDatafeed-{exchange}"
            log_success(f"Fetched {len(df)} candles for {symbol} from {exchange} via tvDatafeed")
            return df, source

        except Exception as e:
            log_error(f"Error processing tvDatafeed result for {symbol}: {e}")
            return None, None

    async def _fetch_ohlcv_tvdatafeed_async(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch OHLCV data using tvDatafeed with multiple exchange fallback (async version).

        Tries exchanges in priority order: OANDA, FXCM, FOREXCOM, IC MARKETS, PEPPERSTONE.

        Args:
            symbol: Forex symbol (e.g., 'EUR/USD', 'GBP/USD')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch (default: 500)

        Returns:
            Tuple[pd.DataFrame, str]: DataFrame with OHLCV data and source string.
            Returns (None, None) if data cannot be fetched from any exchange.
        """
        if not self._tvdatafeed_available:
            return None, None

        interval = self._convert_timeframe_to_tvdatafeed(timeframe)
        if interval is None:
            log_error(f"Could not convert timeframe {timeframe} for tvDatafeed")
            return None, None

        # Try each exchange in priority order
        for exchange in self.SUPPORTED_FOREX_EXCHANGES:
            try:
                # Convert symbol for this exchange
                tv_symbol, normalized_exchange = self._convert_forex_symbol_for_tvdatafeed(symbol, exchange)

                log_info(f"Trying tvDatafeed {normalized_exchange}:{tv_symbol} ({timeframe}, {limit} candles)")

                # Fetch data with timeout
                df = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._throttled_call,
                        self.tv_datafeed.get_hist,
                        symbol=tv_symbol,
                        exchange=normalized_exchange,
                        interval=interval,
                        n_bars=limit,
                        fut_contract=None,
                        extended_session=False,
                    ),
                    timeout=30.0,
                )

                if df is not None and not df.empty:
                    log_success(f"Successfully fetched data from {normalized_exchange}:{tv_symbol} via tvDatafeed")
                    return self._process_tvdatafeed_result(df, symbol, normalized_exchange, limit)
                else:
                    log_warn(f"No data from {normalized_exchange}:{tv_symbol}, trying next exchange...")
                    continue

            except asyncio.TimeoutError:
                log_warn(f"Timeout fetching from {exchange}:{symbol}, trying next exchange...")
                continue
            except Exception as e:
                log_warn(f"Error fetching from {exchange}:{symbol} - {e}, trying next exchange...")
                continue

        log_error(f"Failed to fetch data for {symbol} from all exchanges")
        return None, None

    async def _fetch_tradingview_scraper_async(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch OHLCV data using tradingview_scraper with multiple exchange fallback (async version).

        Args:
            symbol: Forex symbol (e.g., 'EUR/USD', 'GBP/USD')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch (default: 500)

        Returns:
            Tuple[pd.DataFrame, str]: DataFrame with OHLCV data and source string.
            Returns (None, None) if data cannot be fetched from any exchange.
        """
        if not self._tradingview_scraper_available:
            return None, None

        for exchange in self.SUPPORTED_FOREX_EXCHANGES:
            try:
                exchange_symbol = self._convert_forex_symbol_to_tradingview(symbol, exchange)
                self._convert_timeframe_to_tradingview(timeframe)

                log_info(f"Trying tradingview_scraper {exchange}: {exchange_symbol} ({timeframe}, {limit} candles)")

                ohlcv_data = []
                count = 0
                timeout_seconds = 15
                last_data_time = time.time()
                first_candle_received = False

                async def collect_data():
                    nonlocal ohlcv_data, count, last_data_time, first_candle_received
                    for candle in self.real_time_data.get_ohlcv(exchange_symbol=exchange_symbol):
                        if count >= limit:
                            break

                        current_time = time.time()

                        if not first_candle_received and (current_time - last_data_time) > timeout_seconds:
                            break

                        if candle and isinstance(candle, (list, tuple)) and len(candle) >= 5:
                            first_candle_received = True
                            last_data_time = current_time
                            if len(candle) >= 6:
                                ohlcv_data.append(candle[:6])
                            else:
                                ohlcv_data.append(candle)
                            count += 1

                        if (current_time - last_data_time) > timeout_seconds:
                            break

                try:
                    await asyncio.wait_for(collect_data(), timeout=timeout_seconds * 2)
                except asyncio.TimeoutError:
                    log_warn(f"Timeout fetching from {exchange}, trying next exchange...")
                    continue

                if ohlcv_data:
                    df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    df.set_index("timestamp", inplace=True)
                    df.sort_index(inplace=True)

                    required_cols = ["open", "high", "low", "close", "volume"]
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if not missing_cols:
                        source = f"tradingview_scraper-{exchange}"
                        log_success(f"Fetched {len(df)} candles for {symbol} from {exchange} via tradingview_scraper")
                        return df, source
                    else:
                        log_warn(f"{exchange} data missing columns: {missing_cols}, trying next exchange...")
                        continue

            except Exception as e:
                log_warn(f"tradingview_scraper failed for {exchange}:{symbol}: {e}, trying next exchange...")
                continue

        return None, None

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch OHLCV data for forex symbol using tvDatafeed with tradingview_scraper fallback (async version).

        Priority order:
        1. tvDatafeed (primary - more reliable for forex) - tries multiple exchanges
        2. tradingview_scraper (fallback) - tries multiple exchanges

        Args:
            symbol: Forex symbol (e.g., 'EUR/USD', 'GBP/USD')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch (default: 500)

        Returns:
            Tuple[pd.DataFrame, str]: DataFrame with OHLCV data and source string.
            Returns (None, None) if data cannot be fetched from any source.
        """
        if self._tvdatafeed_available:
            df, source = await self._fetch_ohlcv_tvdatafeed_async(symbol, timeframe, limit)
            if df is not None and not df.empty:
                return df, source
            else:
                log_warn(f"tvDatafeed failed for {symbol} from all exchanges, trying tradingview_scraper fallback...")

        if self._tradingview_scraper_available:
            df, source = await self._fetch_tradingview_scraper_async(symbol, timeframe, limit)
            if df is not None and not df.empty:
                return df, source

        log_error(f"Failed to fetch data for {symbol} from both tvDatafeed and tradingview_scraper (all exchanges)")
        return None, None

    def _fetch_ohlcv_tvdatafeed(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch OHLCV data using tvDatafeed with multiple exchange fallback.

        Tries exchanges in priority order: OANDA, FXCM, FOREXCOM, IC MARKETS, PEPPERSTONE.

        Args:
            symbol: Forex symbol (e.g., 'EUR/USD', 'GBP/USD')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch (default: 500)

        Returns:
            Tuple[pd.DataFrame, str]: DataFrame with OHLCV data and source string.
            Returns (None, None) if data cannot be fetched from any exchange.
        """
        if not self._tvdatafeed_available:
            return None, None

        interval = self._convert_timeframe_to_tvdatafeed(timeframe)
        if interval is None:
            log_error(f"Could not convert timeframe {timeframe} for tvDatafeed")
            return None, None

        # Try each exchange in priority order
        last_error = None
        for exchange in self.SUPPORTED_FOREX_EXCHANGES:
            try:
                # Convert symbol for this exchange
                tv_symbol, normalized_exchange = self._convert_forex_symbol_for_tvdatafeed(symbol, exchange)

                log_info(f"Trying tvDatafeed {normalized_exchange}:{tv_symbol} ({timeframe}, {limit} candles)")

                # Fetch data using tvDatafeed.get_hist with signature:
                # get_hist(symbol: str, exchange: str = 'NSE',
                #          interval: Interval = Interval.in_daily,
                #          n_bars: int = 10, fut_contract: int | None = None,
                #          extended_session: bool = False) -> DataFrame
                # For forex: exchange='OANDA'/'FXCM'/'FOREXCOM'/'ICMARKETS'/'PEPPERSTONE',
                # symbol=EURUSD (without '/')
                df = self._throttled_call(
                    self.tv_datafeed.get_hist,
                    symbol=tv_symbol,
                    exchange=normalized_exchange,
                    interval=interval,
                    n_bars=limit,
                    fut_contract=None,
                    extended_session=False,
                )

                if df is not None and not df.empty:
                    # Success! Process and return data
                    log_success(f"Successfully fetched data from {normalized_exchange}:{tv_symbol} via tvDatafeed")
                    return self._process_tvdatafeed_result(df, symbol, normalized_exchange, limit)
                else:
                    log_warn(f"No data from {normalized_exchange}:{tv_symbol}, trying next exchange...")
                    continue

            except Exception as e:
                last_error = e
                log_warn(f"Error fetching from {exchange}:{symbol} - {e}, trying next exchange...")
                continue

        # All exchanges failed
        log_error(f"Failed to fetch data for {symbol} from all exchanges: {last_error}")
        return None, None

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch OHLCV data for forex symbol using tvDatafeed with tradingview_scraper fallback.

        Priority order:
        1. tvDatafeed (primary - more reliable for forex) - tries multiple exchanges
        2. tradingview_scraper (fallback) - tries multiple exchanges

        Args:
            symbol: Forex symbol (e.g., 'EUR/USD', 'GBP/USD')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch (default: 500)

        Returns:
            Tuple[pd.DataFrame, str]: DataFrame with OHLCV data and source string.
            Returns (None, None) if data cannot be fetched from any source.
        """
        # Try tvDatafeed first (more reliable for forex) - tries multiple exchanges
        if self._tvdatafeed_available:
            df, source = self._fetch_ohlcv_tvdatafeed(symbol, timeframe, limit)
            if df is not None and not df.empty:
                return df, source
            else:
                log_warn(f"tvDatafeed failed for {symbol} from all exchanges, trying tradingview_scraper fallback...")

        # Fallback to tradingview_scraper (try exchanges in priority order)
        if self._tradingview_scraper_available:
            for exchange in self.SUPPORTED_FOREX_EXCHANGES:
                try:
                    # Convert symbol to TradingView format with this exchange
                    exchange_symbol = self._convert_forex_symbol_to_tradingview(symbol, exchange)
                    # tv_timeframe = self._convert_timeframe_to_tradingview(timeframe)

                    log_info(f"Trying tradingview_scraper {exchange}: {exchange_symbol} ({timeframe}, {limit} candles)")

                    # Get data generator from TradingView
                    data_generator = self.real_time_data.get_ohlcv(exchange_symbol=exchange_symbol)

                    # Collect data from generator with timeout
                    # tradingview_scraper can hang waiting for data, so we need aggressive timeout
                    ohlcv_data = []
                    count = 0
                    max_attempts = limit * 2  # Safety limit
                    timeout_seconds = 15  # Timeout after 15 seconds without any data
                    start_time = time.time()
                    last_data_time = start_time
                    first_candle_received = False

                    try:
                        for candle in data_generator:
                            current_time = time.time()
                            elapsed = current_time - start_time

                            # If we've been waiting too long without ANY data, try next exchange
                            if not first_candle_received and elapsed > timeout_seconds:
                                log_warn(
                                    f"Timeout: No data received after {timeout_seconds}s from "
                                    f"{exchange}, trying next exchange..."
                                )
                                break

                            # Validate candle data format
                            if candle and isinstance(candle, (list, tuple)) and len(candle) >= 5:
                                first_candle_received = True
                                last_data_time = current_time
                                # Validate candle data format
                                if len(candle) >= 6:
                                    ohlcv_data.append(candle[:6])
                                elif len(candle) == 5:
                                    ohlcv_data.append(candle)
                                count += 1

                                # If we got some data but then stopped, also check timeout
                                if count > 0 and current_time - last_data_time > timeout_seconds:
                                    log_warn(
                                        f"Timeout: Stopped receiving data after {count} candles "
                                        f"from {exchange}, trying next exchange..."
                                    )
                                    break

                            # Check total timeout (double timeout for overall operation)
                            if elapsed > timeout_seconds * 2:
                                log_warn(
                                    f"Total timeout: Exceeded {timeout_seconds * 2}s waiting "
                                    f"for {exchange}, trying next exchange..."
                                )
                                break

                            if count >= limit:
                                break
                            if count >= max_attempts:
                                log_warn(f"Reached max attempts ({max_attempts}) for {exchange_symbol}")
                                break

                    except StopIteration:
                        pass
                    except Exception as e:
                        log_warn(
                            f"Error iterating TradingView data generator for {exchange}: {e}, trying next exchange..."
                        )
                        break  # Break inner loop, continue to next exchange

                    if ohlcv_data:
                        # Convert to DataFrame
                        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

                        # Convert timestamp to datetime
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

                        # Set timestamp as index
                        df.set_index("timestamp", inplace=True)
                        df.sort_index(inplace=True)

                        # Ensure we have required columns
                        required_cols = ["open", "high", "low", "close", "volume"]
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if not missing_cols:
                            source = f"tradingview_scraper-{exchange}"
                            log_success(
                                f"Fetched {len(df)} candles for {symbol} from {exchange} via tradingview_scraper"
                            )
                            return df, source
                        else:
                            log_warn(f"{exchange} data missing columns: {missing_cols}, trying next exchange...")
                            continue  # Try next exchange
                    else:
                        log_warn(f"No data from {exchange}, trying next exchange...")
                        continue  # Try next exchange

                except Exception as e:
                    log_warn(f"tradingview_scraper failed for {exchange}:{symbol}: {e}, trying next exchange...")
                    continue  # Try next exchange

        # Both methods failed
        log_error(f"Failed to fetch data for {symbol} from both tvDatafeed and tradingview_scraper (all exchanges)")
        return None, None

    def fetch_ohlcv_with_fallback(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        _use_tradingview: bool = True,  # noqa: ARG002
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch OHLCV data with automatic fallback.

        This method automatically tries tvDatafeed first,
        then falls back to tradingview_scraper if needed.

        Args:
            symbol: Forex symbol (e.g., 'EUR/USD', 'GBP/USD')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch (default: 500)
            _use_tradingview: Parameter kept for compatibility (not used)

        Returns:
            Tuple[pd.DataFrame, str]: DataFrame with OHLCV data and source string.
            Returns (None, None) if data cannot be fetched from any source.
        """
        # fetch_ohlcv already has automatic fallback built-in (tvDatafeed first, then tradingview_scraper)
        return self.fetch_ohlcv(symbol, timeframe, limit)
