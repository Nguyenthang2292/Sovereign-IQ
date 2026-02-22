"""Metrics calculation for sampling strategies."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from modules.common.ui.logging import log_warn


def calculate_volatility_and_spread(
    symbols: List[str],
    data_fetcher,
    timeframe: str = "1d",
    lookback: int = 14,
    use_rust: bool = True,
    ohlcv_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate volatility (ATR) and spread metrics for symbols.

    Volatility is measured as Average True Range (ATR) normalized by price.
    Spread is measured as (high - low) / close percentage.

    Args:
        symbols: List of symbols to calculate metrics for
        data_fetcher: DataFetcher instance for fetching OHLCV data
        timeframe: Timeframe for data (default: "1d" for daily)
        lookback: Lookback period for ATR calculation (default: 14)
        use_rust: Whether to use Rust backend for faster computation (default: True)
        ohlcv_cache: Optional cache of OHLCV data to avoid re-fetching

    Returns:
        Tuple of (volatility_dict, spread_dict) mapping symbol to metric value
    """
    from modules.common.ui.logging import log_error, log_info, log_success

    volatility = {}
    spread = {}

    # Prepare OHLCV data
    ohlcv_data = {}
    symbols_to_fetch = []

    # First check cache
    if ohlcv_cache:
        for symbol in symbols:
            if symbol in ohlcv_cache:
                df = ohlcv_cache[symbol]
                if df is not None and len(df) >= lookback:
                    ohlcv_data[symbol] = {
                        "high": df["high"].values.astype(np.float64),
                        "low": df["low"].values.astype(np.float64),
                        "close": df["close"].values.astype(np.float64),
                    }
            else:
                symbols_to_fetch.append(symbol)
    else:
        symbols_to_fetch = symbols

    # Fetch remaining symbols if any
    if symbols_to_fetch and data_fetcher:
        # Try Rust implementation first if enabled
        if use_rust:
            try:
                import atc_rust

                # Batch fetch remaining OHLCV data
                for symbol in symbols_to_fetch:
                    try:
                        df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                            symbol, timeframe=timeframe, limit=lookback * 2
                        )
                        if df is not None and len(df) >= lookback:
                            ohlcv_data[symbol] = {
                                "high": df["high"].values.astype(np.float64),
                                "low": df["low"].values.astype(np.float64),
                                "close": df["close"].values.astype(np.float64),
                            }
                    except Exception as e:
                        log_warn(f"[Liquidity Metrics] Failed to cache data for {symbol}: {e}")
                        continue
            except ImportError:
                # If rust not available, we proceed to Python fallback for remaining
                pass
            except Exception as e:
                log_error(f"[Liquidity Metrics] Rust calculation failed: {e}, falling back to Python")

    # If we have data (from cache or fetch), calculate metrics
    if use_rust and ohlcv_data:
        try:
            import atc_rust

            # Call Rust batch function for volatility/spread calculation
            results = atc_rust.compute_liquidity_metrics_batch(ohlcv_data, lookback)
            volatility = results.get("volatility", {})
            spread = results.get("spread", {})

            log_success(f"[Rust] Calculated volatility/spread for {len(volatility)} symbols (ATR lookback: {lookback})")
            return volatility, spread
        except Exception as e:
            log_error(f"[Liquidity Metrics] Rust calculation failed: {e}, falling back to Python")

    # Python fallback implementation
    log_info(f"[Python] Calculating volatility/spread for {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            df = None
            if ohlcv_cache and symbol in ohlcv_cache:
                df = ohlcv_cache[symbol]

            if df is None:
                if data_fetcher:
                    # Fetch OHLCV data
                    df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                        symbol, timeframe=timeframe, limit=lookback * 2
                    )

            if df is None or len(df) < lookback:
                continue

            # Calculate True Range: max(high-low, abs(high-close_prev), abs(low-close_prev))
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values

            # Shift close by 1 for previous close
            close_prev = np.roll(close, 1)
            close_prev[0] = np.nan

            tr = np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))

            # Calculate ATR (Average True Range) - simple moving average of TR
            # Skip first NaN value from rolled array
            atr = np.nanmean(tr[-lookback:])

            # Normalize ATR by current price (ATR%)
            current_price = close[-1]
            volatility[symbol] = (atr / current_price * 100.0) if current_price > 0 else 0.0

            # Calculate average spread percentage
            spread_pct = ((high - low) / close * 100.0)[-lookback:]
            spread[symbol] = float(np.nanmean(spread_pct))

        except Exception as e:
            log_error(f"[Liquidity Metrics] Error calculating metrics for {symbol}: {e}")
            continue

    log_success(f"[Python] Calculated volatility/spread for {len(volatility)} symbols (ATR lookback: {lookback})")
    return volatility, spread
