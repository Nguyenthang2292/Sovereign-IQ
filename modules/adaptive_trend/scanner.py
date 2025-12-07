"""ATC Symbol Scanner.

This module provides functions for scanning multiple symbols using
Adaptive Trend Classification (ATC) to find LONG/SHORT signals.

The scanner fetches data for multiple symbols, calculates ATC signals,
and filters results based on signal strength and trend direction.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.common.DataFetcher import DataFetcher

try:
    from modules.common.utils import (
        log_error,
        log_warn,
        log_success,
        log_progress,
    )
except ImportError:
    def log_error(message: str) -> None:
        print(f"[ERROR] {message}")
    
    def log_warn(message: str) -> None:
        print(f"[WARN] {message}")
    
    def log_success(message: str) -> None:
        print(f"[SUCCESS] {message}")
    
    def log_progress(message: str) -> None:
        print(f"[PROGRESS] {message}")

from modules.adaptive_trend.atc import compute_atc_signals
from modules.adaptive_trend.layer1 import trend_sign


def scan_all_symbols(
    data_fetcher: "DataFetcher",
    timeframe: str,
    limit: int,
    ema_len: int,
    hma_len: int,
    wma_len: int,
    dema_len: int,
    lsma_len: int,
    kama_len: int,
    robustness: str,
    lambda_param: float,
    decay: float,
    cutout: int,
    max_symbols: Optional[int] = None,
    min_signal: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scan all futures symbols and filter those with LONG/SHORT signals.
    
    Fetches OHLCV data for multiple symbols, calculates ATC signals for each,
    and returns DataFrames containing symbols with signals above the threshold,
    separated into LONG (trend > 0) and SHORT (trend < 0) signals.
    
    Args:
        data_fetcher: DataFetcher instance for fetching market data.
        timeframe: Timeframe for data (e.g., '1h', '4h', '1d').
        limit: Number of candles to fetch (must be > 0).
        ema_len: EMA length parameter (must be > 0).
        hma_len: HMA length parameter (must be > 0).
        wma_len: WMA length parameter (must be > 0).
        dema_len: DEMA length parameter (must be > 0).
        lsma_len: LSMA length parameter (must be > 0).
        kama_len: KAMA length parameter (must be > 0).
        robustness: Robustness setting ("Narrow", "Medium", or "Wide").
        lambda_param: Lambda parameter for equity calculations (must be finite).
        decay: Decay rate for equity calculations (must be between 0 and 1).
        cutout: Cutout period (must be >= 0).
        max_symbols: Maximum number of symbols to scan (None = all symbols).
        min_signal: Minimum signal strength to include (must be >= 0).
        
    Returns:
        Tuple of two DataFrames:
        - long_signals_df: Symbols with bullish signals (trend > 0), sorted by signal strength
        - short_signals_df: Symbols with bearish signals (trend < 0), sorted by signal strength
        
        Each DataFrame contains columns: symbol, signal, trend, price, exchange.
        
    Raises:
        ValueError: If any parameter is invalid.
        TypeError: If data_fetcher is None or missing required methods.
        AttributeError: If data_fetcher doesn't have required methods.
    """
    # Input validation
    if data_fetcher is None:
        raise ValueError("data_fetcher cannot be None")
    
    # Validate data_fetcher has required methods
    required_methods = ["list_binance_futures_symbols", "fetch_ohlcv_with_fallback_exchange"]
    for method_name in required_methods:
        if not hasattr(data_fetcher, method_name):
            raise AttributeError(
                f"data_fetcher must have method '{method_name}', "
                f"got {type(data_fetcher)}"
            )
    
    if not isinstance(timeframe, str) or not timeframe.strip():
        raise ValueError(f"timeframe must be a non-empty string, got {timeframe}")
    
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {limit}")
    
    # Validate all MA lengths
    ma_lengths = {
        "ema_len": ema_len,
        "hma_len": hma_len,
        "wma_len": wma_len,
        "dema_len": dema_len,
        "lsma_len": lsma_len,
        "kama_len": kama_len,
    }
    for name, length in ma_lengths.items():
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f"{name} must be a positive integer, got {length}")
    
    # Validate robustness
    VALID_ROBUSTNESS = {"Narrow", "Medium", "Wide"}
    if robustness not in VALID_ROBUSTNESS:
        raise ValueError(
            f"robustness must be one of {VALID_ROBUSTNESS}, got {robustness}"
        )
    
    # Validate lambda_param
    if not isinstance(lambda_param, (int, float)) or np.isnan(lambda_param) or np.isinf(lambda_param):
        raise ValueError(f"lambda_param must be a finite number, got {lambda_param}")
    
    # Validate decay
    if not isinstance(decay, (int, float)) or not (0 <= decay <= 1):
        raise ValueError(f"decay must be between 0 and 1, got {decay}")
    
    # Validate cutout
    if not isinstance(cutout, int) or cutout < 0:
        raise ValueError(f"cutout must be a non-negative integer, got {cutout}")
    
    # Validate max_symbols
    if max_symbols is not None and (not isinstance(max_symbols, int) or max_symbols <= 0):
        raise ValueError(f"max_symbols must be a positive integer or None, got {max_symbols}")
    
    # Validate min_signal
    if not isinstance(min_signal, (int, float)) or min_signal < 0:
        raise ValueError(f"min_signal must be a non-negative number, got {min_signal}")

    try:
        log_progress("Fetching futures symbols from Binance...")
        all_symbols = data_fetcher.list_binance_futures_symbols(
            max_candidates=None,  # Get all symbols first
            progress_label="Symbol Discovery",
        )

        if not all_symbols:
            log_error("No symbols found")
            return pd.DataFrame(), pd.DataFrame()

        # Limit symbols if max_symbols specified
        if max_symbols and max_symbols > 0:
            symbols = all_symbols[:max_symbols]
            log_success(
                f"Found {len(all_symbols)} futures symbols, "
                f"scanning first {len(symbols)} symbols"
            )
        else:
            symbols = all_symbols
            log_success(f"Found {len(symbols)} futures symbols")
        
        log_progress(f"Scanning {len(symbols)} symbols for ATC signals...")

        results = []
        total = len(symbols)
        skipped_count = 0
        error_count = 0
        skipped_symbols = []
        
        for idx, symbol in enumerate(symbols, 1):
            try:
                # Fetch OHLCV data
                df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                    symbol,
                    limit=limit,
                    timeframe=timeframe,
                    check_freshness=True,
                )

                if df is None or df.empty:
                    skipped_count += 1
                    skipped_symbols.append(symbol)
                    continue
                
                if "close" not in df.columns:
                    log_warn(f"Symbol {symbol}: DataFrame missing 'close' column, skipping")
                    skipped_count += 1
                    skipped_symbols.append(symbol)
                    continue

                close_prices = df["close"]
                
                # Validate we have enough data
                if len(close_prices) < limit:
                    log_warn(
                        f"Symbol {symbol}: Insufficient data "
                        f"(got {len(close_prices)}, need {limit}), skipping"
                    )
                    skipped_count += 1
                    skipped_symbols.append(symbol)
                    continue
                
                current_price = close_prices.iloc[-1]
                
                # Validate price is valid
                if pd.isna(current_price) or current_price <= 0:
                    log_warn(f"Symbol {symbol}: Invalid price {current_price}, skipping")
                    skipped_count += 1
                    skipped_symbols.append(symbol)
                    continue

                # Calculate ATC signals
                atc_results = compute_atc_signals(
                    prices=close_prices,
                    src=None,
                    ema_len=ema_len,
                    hull_len=hma_len,
                    wma_len=wma_len,
                    dema_len=dema_len,
                    lsma_len=lsma_len,
                    kama_len=kama_len,
                    ema_w=1.0,
                    hma_w=1.0,
                    wma_w=1.0,
                    dema_w=1.0,
                    lsma_w=1.0,
                    kama_w=1.0,
                    robustness=robustness,
                    La=lambda_param,
                    De=decay,
                    cutout=cutout,
                )

                average_signal = atc_results.get("Average_Signal")
                if average_signal is None or average_signal.empty:
                    skipped_count += 1
                    skipped_symbols.append(symbol)
                    continue

                latest_signal = average_signal.iloc[-1]
                
                # Validate signal is not NaN
                if pd.isna(latest_signal):
                    skipped_count += 1
                    skipped_symbols.append(symbol)
                    continue
                
                latest_trend = trend_sign(average_signal)
                latest_trend_value = latest_trend.iloc[-1] if not latest_trend.empty else 0

                # Only include signals above threshold
                if abs(latest_signal) < min_signal:
                    skipped_count += 1
                    continue

                results.append({
                    "symbol": symbol,
                    "signal": latest_signal,
                    "trend": latest_trend_value,
                    "price": current_price,
                    "exchange": exchange_id or "UNKNOWN",
                })

                # Progress update every 10 symbols
                if idx % 10 == 0 or idx == total:
                    log_progress(
                        f"Scanned {idx}/{total} symbols... "
                        f"Found {len(results)} signals, "
                        f"Skipped {skipped_count}, Errors {error_count}"
                    )

            except KeyboardInterrupt:
                log_warn("Scan interrupted by user")
                break
            except Exception as e:
                # Log errors but continue scanning
                error_count += 1
                skipped_symbols.append(symbol)
                log_warn(
                    f"Error processing symbol {symbol}: {type(e).__name__}: {e}. "
                    f"Skipping and continuing..."
                )
                continue

        # Summary logging
        log_progress(
            f"Scan complete: {total} total, {len(results)} signals found, "
            f"{skipped_count} skipped, {error_count} errors"
        )
        
        if skipped_count > 0 and len(skipped_symbols) <= 10:
            log_warn(f"Skipped symbols: {', '.join(skipped_symbols)}")
        elif skipped_count > 10:
            log_warn(f"Skipped {skipped_count} symbols (first 10: {', '.join(skipped_symbols[:10])}...)")

        if not results:
            log_warn("No signals found above threshold")
            return pd.DataFrame(), pd.DataFrame()

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Filter LONG and SHORT signals
        long_signals = results_df[results_df["trend"] > 0].copy()
        short_signals = results_df[results_df["trend"] < 0].copy()

        # Sort by signal strength (absolute value)
        long_signals = long_signals.sort_values("signal", ascending=False).reset_index(drop=True)
        short_signals = short_signals.sort_values("signal", ascending=True).reset_index(drop=True)

        log_success(
            f"Found {len(long_signals)} LONG signals and {len(short_signals)} SHORT signals"
        )

        return long_signals, short_signals

    except KeyboardInterrupt:
        log_warn("Scan interrupted by user")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        log_error(f"Fatal error scanning symbols: {type(e).__name__}: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), pd.DataFrame()

