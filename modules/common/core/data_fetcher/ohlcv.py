"""OHLCV data fetching with exchange fallback."""

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
    ):
        """
        Fetches OHLCV data using ccxt with fallback exchanges (with caching).

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Number of candles to fetch (default: 1500)
            timeframe: Timeframe string (e.g., '1h', '1d') (default: '1h')
            check_freshness: If True, checks data freshness and tries multiple exchanges (default: False)
            exchanges: Optional list of exchange IDs to try. If None, uses exchange_manager's priority list

        Returns:
            Tuple[pd.DataFrame, str]: DataFrame contains full OHLCV data with columns
            ['timestamp', 'open', 'high', 'low', 'close', 'volume'] and exchange_id string.
            Returns (None, None) if data cannot be fetched.
        """
        normalized_symbol = normalize_symbol(symbol)
        cache_key = (normalized_symbol.upper(), timeframe, int(limit))

        # Check cache (only if not checking freshness, as freshness requires fresh data)
        if not check_freshness:
            if cache_key in self.base._ohlcv_dataframe_cache:
                cached_df, cached_exchange = self.base._ohlcv_dataframe_cache[cache_key]
                # Validate cached DataFrame has required columns
                required_cols = ["high", "low", "close"]
                if cached_df is not None and not cached_df.empty:
                    missing_cols = [col for col in required_cols if col not in cached_df.columns]
                    if not missing_cols:
                        return cached_df.copy(), cached_exchange
                    # Cache has invalid data, remove it and fetch fresh
                    del self.base._ohlcv_dataframe_cache[cache_key]
                elif cached_df is None or cached_df.empty:
                    # Cache has invalid data, remove it and fetch fresh
                    del self.base._ohlcv_dataframe_cache[cache_key]

        # Determine which exchanges to try
        exchange_list = exchanges or self.base.exchange_manager.public.exchange_priority_for_fallback

        # Freshness checking setup
        freshness_minutes = None
        fallback = None
        if check_freshness:
            freshness_minutes = max(timeframe_to_minutes(timeframe) * 1.5, 5)
            log_data(f"Fetching {limit} candles for {normalized_symbol} ({timeframe})...")

        last_error = None
        for exchange_id in exchange_list:
            if self.base.should_stop():
                log_warn("OHLCV fetch cancelled by shutdown.")
                return None, None

            exchange_id = exchange_id.strip()
            if not exchange_id:
                continue

            try:
                # Use public manager for public OHLCV data (no credentials needed)
                exchange = self.base.exchange_manager.public.connect_to_exchange_with_no_credentials(exchange_id)
            except Exception as exc:
                last_error = exc
                if check_freshness:
                    log_warn(f"[{exchange_id.upper()}] Error connecting: {exc}")
                continue

            try:
                # Use public manager's throttled_call
                ohlcv = self.base.exchange_manager.public.throttled_call(
                    exchange.fetch_ohlcv,
                    normalized_symbol,
                    timeframe=timeframe,
                    limit=limit,
                )
            except Exception as exc:
                last_error = exc
                if check_freshness:
                    log_error(f"[{exchange_id.upper()}] Error fetching data: {exc}")
                continue

            if not ohlcv:
                last_error = ValueError(f"{exchange_id}: empty OHLCV")
                if check_freshness:
                    log_warn(f"[{exchange_id.upper()}] No data retrieved.")
                continue

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            if df.empty:
                last_error = ValueError(f"{exchange_id}: OHLCV dataframe empty")
                if check_freshness:
                    log_warn(f"[{exchange_id.upper()}] No data retrieved.")
                continue

            # Convert timestamp and ensure ordering
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

            # Set timestamp as index (DatetimeIndex) to avoid warnings in downstream modules
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            # Check freshness if requested
            if check_freshness:
                last_ts = df.index[-1]  # Use index instead of timestamp column
                now = pd.Timestamp.now(tz="UTC")
                age_minutes = (now - last_ts).total_seconds() / 60.0

                if age_minutes <= freshness_minutes:
                    log_success(f"[{exchange_id.upper()}] Data age {age_minutes:.1f}m (fresh).")
                    self.base._ohlcv_dataframe_cache[cache_key] = (df.copy(), exchange_id)
                    # When check_freshness=True, always return tuple
                    return df, exchange_id

                log_warn(f"[{exchange_id.upper()}] Data age {age_minutes:.1f}m (stale). Trying next exchange...")
                fallback = (df, exchange_id)
                continue

            # No freshness check - use first successful result
            log_success(f"[OHLCV] {normalized_symbol} loaded from {exchange_id} ({len(df)} bars)")

            self.base._ohlcv_dataframe_cache[cache_key] = (df.copy(), exchange_id)
            return df, exchange_id

        # Handle fallback for stale data
        if check_freshness and fallback:
            df, exchange_id = fallback
            log_info(f"Using latest available data from {exchange_id.upper()} despite staleness.")
            self.base._ohlcv_dataframe_cache[cache_key] = (df.copy(), exchange_id)
            # When check_freshness=True, always return tuple
            return df, exchange_id

        # Failed to fetch
        log_error(f"Failed to fetch OHLCV for {normalized_symbol}: {last_error}")
        return None, None
