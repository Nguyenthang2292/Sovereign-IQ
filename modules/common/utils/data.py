"""
Data utilities for DataFrame/Series manipulation and OHLCV data fetching.

This module provides utilities for:
- Validating OHLCV DataFrames
- Transforming DataFrames (e.g., extracting close prices)
- Fetching OHLCV data from exchanges
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from modules.common.core.exchange_manager import ExchangeManager
from modules.common.ui.logging import log_debug, log_error, log_warn

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher


# Constants
OHLCV_REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


# ============================================================================
# Validation Functions
# ============================================================================


def validate_ohlcv_input(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that DataFrame has required columns and is not empty.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """
    if df.empty:
        raise ValueError(f"DataFrame is empty. Required columns: {required_columns}")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")


def validate_price_series(high: pd.Series, low: pd.Series, close: pd.Series) -> None:
    """
    Validate high, low, close price series have correct types and alignment.

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Raises:
        TypeError: If any of the inputs is not a pandas Series
        ValueError: If any series is empty or indices don't match
    """
    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series) or not isinstance(close, pd.Series):
        raise TypeError("high, low, and close must be pandas Series")

    if high.empty or low.empty or close.empty:
        raise ValueError("high, low, and close series cannot be empty")

    if not high.index.equals(low.index) or not low.index.equals(close.index):
        raise ValueError("high, low, and close must have the same index")


def _validate_symbol_and_timeframe(symbol: str, timeframe: str) -> bool:
    """
    Validate symbol and timeframe are non-empty strings.

    Args:
        symbol: Symbol to validate
        timeframe: Timeframe to validate

    Returns:
        True if both are valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        log_warn(f"Skipping invalid symbol: {symbol}")
        return False
    if not timeframe or not isinstance(timeframe, str):
        log_warn(f"Skipping invalid timeframe '{timeframe}' for symbol '{symbol}'")
        return False
    return True


# ============================================================================
# Transformation Functions
# ============================================================================


def dataframe_to_close_series(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """
    Converts a fetched OHLCV DataFrame into a pandas Series of closing prices indexed by timestamp.

    Args:
        df: OHLCV DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    Returns:
        pandas Series of closing prices indexed by timestamp, or None if input is invalid
    """
    if df is None or df.empty:
        return None
    if "timestamp" not in df.columns or "close" not in df.columns:
        return None
    series = df.set_index("timestamp")["close"].copy()
    series.name = "close"
    return series


def _normalize_dataframe_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has 'timestamp' column.

    If timestamp is in index (DatetimeIndex), reset it to column.

    Args:
        df: DataFrame to normalize

    Returns:
        DataFrame with 'timestamp' column
    """
    if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    return df


# ============================================================================
# Data Fetching Functions
# ============================================================================


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
