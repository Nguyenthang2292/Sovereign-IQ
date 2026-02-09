"""OHLCV data fetching with exchange fallback."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Optional, Tuple

import pandas as pd

from modules.common.data import dataframe_to_close_series
from modules.common.domain import normalize_symbol, timeframe_to_minutes
from modules.common.ui.logging import log_data, log_error, log_info, log_success, log_warn

if TYPE_CHECKING:
    from .base import DataFetcherBase


class OHLCVFetcher:
    """Handles OHLCV data fetching with exchange fallback."""

    def __init__(self, base: "DataFetcherBase"):
        """
        Initialize OHLCVFetcher.

        Args:
            base: DataFetcherBase instance for accessing exchange_manager and state
        """
        self.base = base

    @staticmethod
    def dataframe_to_close_series(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
        """
        Converts a fetched OHLCV DataFrame into a pandas Series of closing prices indexed by timestamp.

        This is a wrapper method for backward compatibility. The actual implementation
        is in modules.common.utils.data.dataframe_to_close_series().

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series of closing prices indexed by timestamp, or None if input is None/empty
        """
        return dataframe_to_close_series(df)

    def fetch_ohlcv_with_fallback_exchange(
        self,
        symbol,
        limit=1500,
        timeframe="1h",
        check_freshness=False,
        exchanges=None,
        cache_ttl_multiplier=1.0,
    ):
        """
        Fetches OHLCV data using ccxt with fallback exchanges (with caching).

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Number of candles to fetch (default: 1500)
            timeframe: Timeframe string (e.g., '1h', '1d') (default: '1h')
            check_freshness: If True, checks data freshness and tries multiple exchanges (default: False)
            exchanges: Optional list of exchange IDs to try. If None, uses exchange_manager's priority list
            cache_ttl_multiplier: Multiplier for cache TTL when check_freshness=True (default: 1.0 = 1x timeframe)

        Returns:
            Tuple[pd.DataFrame, str]: DataFrame contains full OHLCV data with columns
            ['timestamp', 'open', 'high', 'low', 'close', 'volume'] and exchange_id string.
            Returns (None, None) if data cannot be fetched.
        """
        normalized_symbol = normalize_symbol(symbol)
        cache_key = (normalized_symbol.upper(), timeframe, int(limit))

        # Calculate TTL based on timeframe
        ttl_seconds = timeframe_to_minutes(timeframe) * 60 * cache_ttl_multiplier

        # Check cache with TTL logic
        if cache_key in self.base._ohlcv_dataframe_cache:
            cached_df, cached_exchange = self.base._ohlcv_dataframe_cache[cache_key]
            cache_timestamp = self.base._ohlcv_cache_timestamps.get(cache_key, 0)

            # Validate cached DataFrame has required columns
            required_cols = ["high", "low", "close"]
            if cached_df is not None and not cached_df.empty:
                missing_cols = [col for col in required_cols if col not in cached_df.columns]
                if not missing_cols:
                    cache_age_seconds = time.time() - cache_timestamp

                    # If check_freshness=True, only return cached if within TTL
                    if check_freshness:
                        if cache_age_seconds < ttl_seconds:
                            log_info(
                                f"[CACHE] {normalized_symbol} ({timeframe}) - served from cache (age: {cache_age_seconds:.0f}s)"
                            )
                            return cached_df.copy(), cached_exchange
                        log_info(
                            f"[CACHE] {normalized_symbol} ({timeframe}) - cache expired (age: {cache_age_seconds:.0f}s), fetching fresh"
                        )
                    else:
                        # Normal mode: always return cached if valid
                        return cached_df.copy(), cached_exchange

            # Cache has invalid data, remove it and fetch fresh
            del self.base._ohlcv_dataframe_cache[cache_key]
            if cache_key in self.base._ohlcv_cache_timestamps:
                del self.base._ohlcv_cache_timestamps[cache_key]

        # Determine which exchanges to try
        exchange_list = exchanges or self.base.exchange_manager.public.exchange_priority_for_fallback

        # Freshness checking setup
        freshness_minutes = None
        fallback = None
        if check_freshness:
            freshness_minutes = max(timeframe_to_minutes(timeframe) * 1.5, 5)
            log_data(f"Fetching {limit} candles for {normalized_symbol} ({timeframe})...")

        # Try exchanges - first N in parallel, then sequential fallback
        return self._try_exchanges_parallel(
            exchange_list, normalized_symbol, timeframe, limit, cache_key, check_freshness, freshness_minutes
        )

    def _try_exchanges_parallel(
        self,
        exchange_list: list,
        symbol: str,
        timeframe: str,
        limit: int,
        cache_key: Tuple,
        check_freshness: bool,
        freshness_minutes: Optional[float],
        parallel_probe_count: int = 3,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Try exchanges: probe first N in parallel, then fall back to sequential.

        Returns (df, exchange_id) on success, (None, None) on failure.
        """
        fallback = None
        last_error = None

        # Phase 1: Probe first N exchanges in parallel
        probe_exchanges = [ex.strip() for ex in exchange_list[:parallel_probe_count] if ex.strip()]
        remaining_exchanges = [ex.strip() for ex in exchange_list[parallel_probe_count:] if ex.strip()]

        if probe_exchanges:
            log_info(f"[PARALLEL] Probing {len(probe_exchanges)} exchanges in parallel: {probe_exchanges}")

            with ThreadPoolExecutor(max_workers=len(probe_exchanges)) as executor:
                # Submit all probe tasks
                future_to_exchange = {
                    executor.submit(self._fetch_from_exchange, ex, symbol, timeframe, limit, check_freshness): ex
                    for ex in probe_exchanges
                }

                # Process results as they complete (first successful wins)
                for future in as_completed(future_to_exchange):
                    if self.base.should_stop():
                        log_warn("OHLCV fetch cancelled by shutdown.")
                        return None, None

                    exchange_id = future_to_exchange[future]
                    try:
                        result = future.result()
                        if result is not None:
                            df, error = result
                            if df is not None:
                                # Validate and check freshness
                                validated_df = self._validate_and_process_df(df, check_freshness, freshness_minutes)
                                if validated_df is not None:
                                    log_success(f"[PARALLEL] Success from {exchange_id.upper()}")
                                    self.base._ohlcv_dataframe_cache[cache_key] = (validated_df.copy(), exchange_id)
                                    self.base._ohlcv_cache_timestamps[cache_key] = time.time()
                                    return validated_df, exchange_id
                                else:
                                    # Data exists but stale, save as fallback
                                    fallback = (df, exchange_id)
                            elif error:
                                last_error = error
                    except Exception as exc:
                        last_error = exc
                        if check_freshness:
                            log_warn(f"[{exchange_id.upper()}] Error in parallel probe: {exc}")

        # Phase 2: Sequential fallback for remaining exchanges
        for exchange_id in remaining_exchanges:
            if self.base.should_stop():
                log_warn("OHLCV fetch cancelled by shutdown.")
                return None, None

            result = self._fetch_from_exchange(exchange_id, symbol, timeframe, limit, check_freshness)
            if result is not None:
                df, error = result
                if df is not None:
                    validated_df = self._validate_and_process_df(df, check_freshness, freshness_minutes)
                    if validated_df is not None:
                        log_success(f"[SEQUENTIAL] Success from {exchange_id.upper()}")
                        self.base._ohlcv_dataframe_cache[cache_key] = (validated_df.copy(), exchange_id)
                        self.base._ohlcv_cache_timestamps[cache_key] = time.time()
                        return validated_df, exchange_id
                    else:
                        fallback = (df, exchange_id)
                elif error:
                    last_error = error

        # Handle fallback for stale data
        if check_freshness and fallback:
            df, exchange_id = fallback
            log_info(f"Using latest available data from {exchange_id.upper()} despite staleness.")
            self.base._ohlcv_dataframe_cache[cache_key] = (df.copy(), exchange_id)
            self.base._ohlcv_cache_timestamps[cache_key] = time.time()
            return df, exchange_id

        # Failed to fetch
        log_error(f"Failed to fetch OHLCV for {symbol}: {last_error}")
        return None, None

    def _fetch_from_exchange(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        limit: int,
        check_freshness: bool,
    ) -> Optional[Tuple[Optional[pd.DataFrame], Optional[Exception]]]:
        """
        Fetch OHLCV from a single exchange.

        Returns (df, None) on success, (None, error) on failure, or None if shutdown requested.
        """
        try:
            exchange = self.base.exchange_manager.public.connect_to_exchange_with_no_credentials(exchange_id)
        except Exception as exc:
            if check_freshness:
                log_warn(f"[{exchange_id.upper()}] Error connecting: {exc}")
            return None, exc

        try:
            ohlcv = self.base.exchange_manager.public.throttled_call(
                exchange.fetch_ohlcv,
                symbol,
                timeframe=timeframe,
                limit=limit,
            )
        except Exception as exc:
            if check_freshness:
                log_error(f"[{exchange_id.upper()}] Error fetching data: {exc}")
            return None, exc

        if not ohlcv:
            if check_freshness:
                log_warn(f"[{exchange_id.upper()}] No data retrieved.")
            return None, ValueError(f"{exchange_id}: empty OHLCV")

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df.empty:
            if check_freshness:
                log_warn(f"[{exchange_id.upper()}] No data retrieved.")
            return None, ValueError(f"{exchange_id}: OHLCV dataframe empty")

        return df, None

    def _validate_and_process_df(
        self,
        df: pd.DataFrame,
        check_freshness: bool,
        freshness_minutes: Optional[float],
    ) -> Optional[pd.DataFrame]:
        """
        Validate DataFrame and check freshness if required.

        Returns processed DataFrame if valid and fresh, None otherwise.
        """
        # Convert timestamp and ensure ordering
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        # Check freshness if requested
        if check_freshness and freshness_minutes is not None:
            last_ts = df.index[-1]
            now = pd.Timestamp.now(tz="UTC")
            age_minutes = (now - last_ts).total_seconds() / 60.0

            if age_minutes > freshness_minutes:
                # Stale data - return None to signal fallback needed
                return None

        return df
