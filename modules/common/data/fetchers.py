"""
Data fetching utilities for OHLCV data from exchanges.

This module provides functions to fetch OHLCV data for multiple symbols and timeframes.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from modules.common.core.exchange_manager import ExchangeManager
from modules.common.ui.logging import log_debug, log_error, log_warn

from .transformation import _normalize_dataframe_timestamp
from .validation import OHLCV_REQUIRED_COLUMNS, _validate_symbol_and_timeframe

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher


def fetch_ohlcv_data_dict(
    symbols: List[str],
    timeframes: List[str],
    exchange_manager: Optional[ExchangeManager] = None,
    data_fetcher: Optional["DataFetcher"] = None,
    limit: int = 1500,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch OHLCV data for multiple symbols and timeframes, with validation and logging.

    Args:
        symbols: List of crypto symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
        timeframes: List of timeframes to load (e.g., ['1h', '4h'])
        exchange_manager: Optional ExchangeManager instance (creates new if None)
        data_fetcher: Optional DataFetcher instance (creates new if None)
        limit: Number of candles to fetch per symbol/timeframe (default: 1500)

    Returns:
        Dictionary with structure {symbol: {timeframe: DataFrame}}
    """
    # Lazy import to avoid circular dependency
    from modules.common.core.data_fetcher import DataFetcher

    if not symbols or not timeframes:
        log_error(f"Symbols or timeframes list is empty. symbols={symbols}, timeframes={timeframes}")
        return {}

    if exchange_manager is None:
        exchange_manager = ExchangeManager()
    if data_fetcher is None:
        data_fetcher = DataFetcher(exchange_manager)

    all_symbols_ohlcv_data = {}
    error_count = 0

    for symbol in symbols:
        symbol_data, symbol_errors = _fetch_symbol_data(
            symbol=symbol, timeframes=timeframes, data_fetcher=data_fetcher, limit=limit
        )
        error_count += symbol_errors

        # Only add symbol_data if it has at least one non-empty DataFrame
        if symbol_data:
            all_symbols_ohlcv_data[symbol] = symbol_data
        else:
            log_warn(f"No usable data loaded for symbol '{symbol}' (all timeframes empty or errored).")

    if error_count:
        log_warn(
            f"fetch_ohlcv_data_dict completed with {error_count} error(s) "
            f"for symbols {symbols} and timeframes {timeframes}"
        )
    if not all_symbols_ohlcv_data:
        log_error("fetch_ohlcv_data_dict did not load any data.")

    return all_symbols_ohlcv_data


def _fetch_symbol_data(
    symbol: str, timeframes: List[str], data_fetcher: "DataFetcher", limit: int
) -> Tuple[Dict[str, pd.DataFrame], int]:
    """
    Fetch OHLCV data for a single symbol across multiple timeframes.

    Args:
        symbol: Crypto symbol (e.g., 'BTCUSDT')
        timeframes: List of timeframes to load
        data_fetcher: DataFetcher instance
        limit: Number of candles to fetch per timeframe

    Returns:
        Tuple of (symbol_data dictionary, error_count)
    """
    symbol_data = {}
    error_count = 0

    for timeframe in timeframes:
        if not _validate_symbol_and_timeframe(symbol, timeframe):
            continue

        df, had_error = _fetch_single_timeframe(
            symbol=symbol, timeframe=timeframe, data_fetcher=data_fetcher, limit=limit
        )

        if df is not None:
            symbol_data[timeframe] = df
        if had_error:
            error_count += 1

    return symbol_data, error_count


def _fetch_single_timeframe(
    symbol: str, timeframe: str, data_fetcher: "DataFetcher", limit: int
) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Fetch OHLCV data for a single symbol/timeframe pair.

    Args:
        symbol: Crypto symbol
        timeframe: Timeframe string
        data_fetcher: DataFetcher instance
        limit: Number of candles to fetch

    Returns:
        Tuple of (DataFrame if successful else None, had_error flag)
    """
    try:
        df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol=symbol, timeframe=timeframe, limit=limit, check_freshness=False
        )

        if df is None or df.empty:
            log_warn(f"Failed to load data for {symbol} {timeframe}")
            return None, False

        # Normalize timestamp column
        df = _normalize_dataframe_timestamp(df)

        # Verify required OHLCV structure
        missing_cols = OHLCV_REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            log_warn(f"Missing columns {missing_cols} in {symbol} {timeframe} from {exchange_id}")
            return None, False

        log_debug(f"Loaded {symbol} {timeframe}: {len(df)} rows from {exchange_id}")
        return df, False

    except Exception as e:
        log_error(f"Error loading data for {symbol} {timeframe}: {e}")
        return None, True


__all__ = ["fetch_ohlcv_data_dict"]
