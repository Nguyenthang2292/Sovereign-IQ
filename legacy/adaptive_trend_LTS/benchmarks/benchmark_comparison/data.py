"""Data fetching utilities for benchmark comparison."""

from typing import Dict

import pandas as pd

from modules.common.core import DataFetcher, ExchangeManager
from modules.common.utils import log_error, log_info, log_success, log_warn


def fetch_symbols_data(num_symbols: int = 1000, bars: int = 1000, timeframe: str = "1h") -> Dict[str, pd.Series]:
    """Fetch price data for multiple symbols.

    Args:
        num_symbols: Number of symbols to fetch (default: 1000)
        bars: Number of bars per symbol (default: 1000)
        timeframe: Timeframe for OHLCV data (default: "1h")

    Returns:
        Dictionary mapping symbol -> close price Series
    """
    log_info(f"Fetching {num_symbols} symbols with {bars} bars each...")

    exchange_mgr = ExchangeManager()
    data_fetcher = DataFetcher(exchange_mgr)

    # Get list of symbols
    log_info("Discovering symbols from Binance Futures...")
    symbols = data_fetcher.list_binance_futures_symbols(max_candidates=num_symbols)

    if len(symbols) < num_symbols:
        log_warn(f"Only {len(symbols)} symbols available, requested {num_symbols}")

    # Fetch data for each symbol
    prices_data = {}
    successful = 0
    failed = 0

    for idx, symbol in enumerate(symbols[:num_symbols], 1):
        try:
            df, exchange = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol=symbol, limit=bars, timeframe=timeframe
            )

            if df is not None and len(df) >= bars:
                close_series = data_fetcher.dataframe_to_close_series(df)
                prices_data[symbol] = close_series
                successful += 1

                if idx % 50 == 0:
                    log_info(f"Progress: {idx}/{num_symbols} symbols fetched")
            else:
                failed += 1
                log_warn(f"Insufficient data for {symbol}")

        except Exception as e:
            failed += 1
            log_error(f"Error fetching {symbol}: {e}")

    log_success(f"Fetched {successful} symbols successfully, {failed} failed")
    return prices_data
