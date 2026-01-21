"""Symbol processing function for ATC scanner.

This module provides the _process_symbol function to process a single symbol:
fetch data and calculate ATC signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import pandas as pd

from modules.adaptive_trend_enhance.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_enhance.core.process_layer1 import trend_sign
from modules.adaptive_trend_enhance.utils.config import ATCConfig

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher


def _process_symbol(
    symbol: str,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
) -> Optional[Dict[str, Any]]:
    """
    Process a single symbol: fetch data and calculate ATC signals.

    Args:
        symbol: Symbol to process
        data_fetcher: DataFetcher instance
        atc_config: ATCConfig object
        min_signal: Minimum signal strength threshold

    Returns:
        Dictionary with symbol data if signal found, None otherwise
    """
    try:
        # Fetch OHLCV data
        df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=atc_config.limit,
            timeframe=atc_config.timeframe,
            check_freshness=True,
        )

        if df is None or df.empty:
            return None

        # Get price source based on calculation_source config
        calculation_source = atc_config.calculation_source.lower()
        valid_sources = ["close", "open", "high", "low"]

        if calculation_source not in valid_sources:
            calculation_source = "close"

        if calculation_source not in df.columns:
            return None

        # Ensure Series is backed by contiguous numpy array for SIMD optimization
        # Numba performs much faster on contiguous arrays (C-style layout)
        import numpy as np

        # Force a copy to C-contiguous array if not already
        # We replace the underlying values but keep the Series wrapper
        # The computation functions extract .values, so we need .values to be contiguous

        raw_values = df[calculation_source].values
        if not raw_values.flags["C_CONTIGUOUS"]:
            raw_values = np.ascontiguousarray(raw_values)

        price_series = pd.Series(raw_values, index=df.index, name=calculation_source)

        # Validate we have enough data
        if len(price_series) < atc_config.limit:
            return None

        current_price = price_series.iloc[-1]

        # Validate price is valid
        if pd.isna(current_price) or current_price <= 0:
            return None

        # Calculate ATC signals
        atc_results = compute_atc_signals(
            prices=price_series,
            src=None,
            ema_len=atc_config.ema_len,
            hull_len=atc_config.hma_len,
            wma_len=atc_config.wma_len,
            dema_len=atc_config.dema_len,
            lsma_len=atc_config.lsma_len,
            kama_len=atc_config.kama_len,
            ema_w=atc_config.ema_w,
            hma_w=atc_config.hma_w,
            wma_w=atc_config.wma_w,
            dema_w=atc_config.dema_w,
            lsma_w=atc_config.lsma_w,
            kama_w=atc_config.kama_w,
            robustness=atc_config.robustness,
            La=atc_config.lambda_param,
            De=atc_config.decay,
            cutout=atc_config.cutout,
            long_threshold=atc_config.long_threshold,
            short_threshold=atc_config.short_threshold,
        )

        average_signal = atc_results.get("Average_Signal")
        if average_signal is None or average_signal.empty:
            return None

        latest_signal = average_signal.iloc[-1]

        # Validate signal is not NaN
        if pd.isna(latest_signal):
            return None

        latest_trend = trend_sign(average_signal)
        latest_trend_value = latest_trend.iloc[-1] if not latest_trend.empty else 0

        # Only include signals above threshold
        if abs(latest_signal) < min_signal:
            return None

        return {
            "symbol": symbol,
            "signal": latest_signal,
            "trend": latest_trend_value,
            "price": current_price,
            "exchange": exchange_id or "UNKNOWN",
        }
    except Exception:
        # Return None on any error - errors are logged in the calling function
        return None
