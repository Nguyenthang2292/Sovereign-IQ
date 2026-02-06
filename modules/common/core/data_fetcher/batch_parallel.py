"""Parallel batch data fetching for improved performance.

This module provides batch parallel OHLCV data fetching capabilities to improve
performance when fetching data for multiple symbols. It uses ThreadPoolExecutor
to fetch data for multiple symbols concurrently.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from modules.common.ui.logging import log_info, log_warn

if TYPE_CHECKING:
    from .base import DataFetcherBase


class BatchParallelFetcher:
    """Handles parallel batch OHLCV data fetching."""

    def __init__(self, base: "DataFetcherBase"):
        """
        Initialize BatchParallelFetcher.

        Args:
            base: DataFetcherBase instance for accessing exchange_manager and state
        """
        self.base = base

    def fetch_ohlcv_batch_parallel(
        self,
        symbols: List[str],
        limit: int = 1500,
        timeframe: str = "1h",
        check_freshness: bool = False,
        exchanges: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Tuple[Optional[pd.DataFrame], Optional[str]]]:
        """
        Fetch OHLCV data for multiple symbols in parallel.

        This method fetches data for multiple symbols concurrently using ThreadPoolExecutor,
        which is significantly faster than sequential fetching when dealing with many symbols.

        Args:
            symbols: List of trading pair symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
            limit: Number of candles to fetch (default: 1500)
            timeframe: Timeframe string (e.g., '1h', '1d') (default: '1h')
            check_freshness: If True, checks data freshness and tries multiple exchanges (default: False)
            exchanges: Optional list of exchange IDs to try. If None, uses exchange_manager's priority list
            max_workers: Maximum number of worker threads. If None, uses min(32, len(symbols) + 4)

        Returns:
            Dictionary mapping symbol to (DataFrame, exchange_id) tuple.
            DataFrame contains OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
            Returns (None, None) for symbols where data cannot be fetched.

        Example:
            >>> results = fetcher.fetch_ohlcv_batch_parallel(['BTC/USDT', 'ETH/USDT'], limit=1000)
            >>> btc_df, btc_exchange = results['BTC/USDT']
        """
        if not symbols:
            log_warn("fetch_ohlcv_batch_parallel: empty symbols list")
            return {}

        if max_workers is None:
            max_workers = min(32, len(symbols) + 4)

        results: Dict[str, Tuple[Optional[pd.DataFrame], Optional[str]]] = {}

        log_info(f"Fetching OHLCV data for {len(symbols)} symbols in parallel (max_workers={max_workers})...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fetch tasks
            future_to_symbol = {
                executor.submit(
                    self._fetch_single_ohlcv,
                    symbol,
                    limit,
                    timeframe,
                    check_freshness,
                    exchanges,
                ): symbol
                for symbol in symbols
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1

                try:
                    df, exchange_id = future.result()
                    results[symbol] = (df, exchange_id)

                    if completed % 10 == 0 or completed == len(symbols):
                        success_count = sum(1 for df, _ in results.values() if df is not None)
                        log_info(
                            f"Batch fetch progress: {completed}/{len(symbols)} complete "
                            f"({success_count} successful)"
                        )
                except Exception as e:
                    log_warn(f"Error fetching {symbol}: {type(e).__name__}: {e}")
                    results[symbol] = (None, None)

        success_count = sum(1 for df, _ in results.values() if df is not None)
        log_info(f"Batch fetch complete: {success_count}/{len(symbols)} symbols successfully fetched")

        return results

    def _fetch_single_ohlcv(
        self,
        symbol: str,
        limit: int,
        timeframe: str,
        check_freshness: bool,
        exchanges: Optional[List[str]],
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Internal method to fetch OHLCV for a single symbol.

        This is called by fetch_ohlcv_batch_parallel in parallel threads.

        Args:
            symbol: Trading pair symbol
            limit: Number of candles to fetch
            timeframe: Timeframe string
            check_freshness: Whether to check data freshness
            exchanges: Optional list of exchange IDs to try

        Returns:
            Tuple of (DataFrame, exchange_id) or (None, None) on failure
        """
        try:
            # Use the existing fetch_ohlcv_with_fallback_exchange method from OHLCVFetcher
            # This ensures we maintain the same behavior and caching logic
            return getattr(self.base, "_ohlcv").fetch_ohlcv_with_fallback_exchange(
                symbol=symbol,
                limit=limit,
                timeframe=timeframe,
                check_freshness=check_freshness,
                exchanges=exchanges,
            )
        except Exception as e:
            log_warn(f"_fetch_single_ohlcv failed for {symbol}: {type(e).__name__}: {e}")
            return None, None

    def prefetch_symbols_data(
        self,
        symbols: List[str],
        limit: int = 1500,
        timeframe: str = "1h",
        check_freshness: bool = False,
        exchanges: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
    ) -> int:
        """
        Prefetch and cache OHLCV data for multiple symbols.

        This method fetches data for multiple symbols in parallel and caches the results.
        Subsequent calls to fetch_ohlcv_with_fallback_exchange for these symbols will use
        the cached data, improving performance.

        This is useful when you know you'll need data for multiple symbols and want to
        fetch them all at once efficiently.

        Args:
            symbols: List of trading pair symbols
            limit: Number of candles to fetch (default: 1500)
            timeframe: Timeframe string (default: '1h')
            check_freshness: If True, checks data freshness (default: False)
            exchanges: Optional list of exchange IDs to try
            max_workers: Maximum number of worker threads

        Returns:
            Number of symbols successfully fetched and cached

        Example:
            >>> # Prefetch data for all symbols you'll need
            >>> success_count = fetcher.prefetch_symbols_data(all_symbols, limit=1000, max_workers=10)
            >>> # Now individual fetches will use cached data
            >>> df, exchange = fetcher.fetch_ohlcv_with_fallback_exchange('BTC/USDT', limit=1000)
        """
        results = self.fetch_ohlcv_batch_parallel(
            symbols=symbols,
            limit=limit,
            timeframe=timeframe,
            check_freshness=check_freshness,
            exchanges=exchanges,
            max_workers=max_workers,
        )

        # Count successful fetches (data is already cached by fetch_ohlcv_with_fallback_exchange)
        success_count = sum(1 for df, _ in results.values() if df is not None)

        log_info(f"Prefetch complete: {success_count}/{len(symbols)} symbols cached")

        return success_count
