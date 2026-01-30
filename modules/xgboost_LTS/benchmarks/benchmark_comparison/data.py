"""Data fetching utilities for benchmark comparison."""

from typing import Dict

import pandas as pd

from modules.common.core import DataFetcher, ExchangeManager
from modules.common.utils import log_error, log_info, log_success, log_warn

# Minimum bars required for a symbol to be considered valid for benchmarking.
# Exchanges (e.g. Binance via ccxt) often cap OHLCV at 1000–1500 per request.
MIN_BARS_FOR_BENCHMARK = 500


def fetch_symbols_data(num_symbols: int = 1000, bars: int = 5000, timeframe: str = "15m") -> Dict[str, pd.DataFrame]:
    """Fetch price data for multiple symbols.

    Args:
        num_symbols: Number of symbols to fetch (default: 1000)
        bars: Requested bars per symbol (default: 5000). Exchanges may return fewer
            (e.g. Binance caps at 1500); symbols with at least MIN_BARS_FOR_BENCHMARK
            are still accepted.
        timeframe: Timeframe for OHLCV data (default: "15m")

    Returns:
        Dictionary mapping symbol -> DataFrame (OHLCV).
        XGBoost needs the full DataFrame, not just close prices.
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
    symbols_data: Dict[str, pd.DataFrame] = {}
    successful = 0
    failed = 0
    any_short = False

    for idx, symbol in enumerate(symbols[:num_symbols], 1):
        try:
            df, exchange = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol=symbol, limit=bars, timeframe=timeframe
            )

            if df is not None and len(df) >= MIN_BARS_FOR_BENCHMARK:
                symbols_data[symbol] = df
                successful += 1
                if len(df) < bars:
                    any_short = True
                if idx % 10 == 0:
                    log_info(f"Progress: {idx}/{num_symbols} symbols fetched")
            else:
                failed += 1
                if df is not None:
                    log_warn(f"Insufficient data for {symbol}: got {len(df)} bars (min {MIN_BARS_FOR_BENCHMARK})")

        except Exception as e:
            failed += 1
            log_error(f"Error fetching {symbol}: {e}")

    if any_short:
        log_warn(
            f"Some symbols have fewer than {bars} bars (exchange limit). "
            f"Benchmark uses all symbols with >= {MIN_BARS_FOR_BENCHMARK} bars."
        )
    log_success(f"Fetched {successful} symbols successfully, {failed} failed")
    return symbols_data
