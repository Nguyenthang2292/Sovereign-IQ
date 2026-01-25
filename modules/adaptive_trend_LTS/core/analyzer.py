"""
ATC Symbol Analyzer.

This module provides functions for analyzing individual symbols using
Adaptive Trend Classification (ATC).
"""

import traceback
from typing import TYPE_CHECKING, Any, Dict, Optional

from modules.adaptive_trend_enhance.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_enhance.utils.config import ATCConfig
from modules.common.system import get_memory_manager
from modules.common.utils import log_warn

__all__ = ["analyze_symbol"]

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher

try:
    from modules.common.utils import (
        log_error,
        log_progress,
    )
except ImportError:

    def log_error(message: str) -> None:
        print(f"[ERROR] {message}")

    def log_progress(message: str) -> None:
        print(f"[PROGRESS] {message}")


def analyze_symbol(
    symbol: str,
    data_fetcher: "DataFetcher",
    config: ATCConfig,
) -> Optional[Dict[str, Any]]:
    """
    Analyze a single symbol using ATC.

    This function computes ATC signals and returns the results. It does not
    handle display - that should be done by the calling code.

    Args:
        symbol: Symbol to analyze
        data_fetcher: DataFetcher instance
        config: ATCConfig containing all ATC parameters

    Returns:
        Dictionary containing analysis results with keys:
            - symbol: Symbol name
            - df: OHLCV DataFrame
            - atc_results: ATC signals dictionary
            - current_price: Current price
            - exchange_label: Exchange identifier
        Returns None if analysis failed.
    """
    mem_manager = get_memory_manager()
    with mem_manager.safe_memory_operation(f"analyze_symbol:{symbol}"):
        try:
            # Fetch OHLCV data
            df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=config.limit,
                timeframe=config.timeframe,
                check_freshness=True,
            )

            if df is None or df.empty:
                log_error(f"No data available for {symbol}")
                return None

            exchange_label = exchange_id.upper() if exchange_id else "UNKNOWN"

            # Get price source based on calculation_source config
            calculation_source = config.calculation_source.lower()
            valid_sources = ["close", "open", "high", "low"]

            if calculation_source not in valid_sources:
                log_warn(f"Invalid calculation_source '{calculation_source}', using 'close'")
                calculation_source = "close"

            if calculation_source not in df.columns:
                log_error(f"No '{calculation_source}' column in data for {symbol}")
                return None

            price_series = df[calculation_source]
            current_price = price_series.iloc[-1]

            # Calculate ATC signals
            log_progress(f"Calculating ATC signals for {symbol} using {calculation_source} prices...")

            atc_results = compute_atc_signals(
                prices=price_series,
                src=None,  # Use selected price source
                ema_len=config.ema_len,
                hull_len=config.hma_len,
                wma_len=config.wma_len,
                dema_len=config.dema_len,
                lsma_len=config.lsma_len,
                kama_len=config.kama_len,
                ema_w=config.ema_w,
                hma_w=config.hma_w,
                wma_w=config.wma_w,
                dema_w=config.dema_w,
                lsma_w=config.lsma_w,
                kama_w=config.kama_w,
                robustness=config.robustness,
                La=config.lambda_param,
                De=config.decay,
                cutout=config.cutout,
                long_threshold=config.long_threshold,
                short_threshold=config.short_threshold,
                parallel_l1=config.parallel_l1,
                parallel_l2=config.parallel_l2,
                use_rust_backend=config.use_rust_backend,
            )

            # Return results instead of displaying
            return {
                "symbol": symbol,
                "df": df,
                "atc_results": atc_results,
                "current_price": current_price,
                "exchange_label": exchange_label,
            }

        except Exception as e:
            log_error(f"Error analyzing {symbol}: {type(e).__name__}: {e}")
            log_error(f"Traceback: {traceback.format_exc()}")
            return None
