"""Pre-filter runner for batch scanner."""

from typing import List, Optional

from modules.common.core.data_fetcher import DataFetcher, SymbolFetchError
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.ui.logging import log_info
from modules.common.utils import log_error, log_warn
from modules.gemini_chart_analyzer.core.prefilter.legacy_voting import (
    pre_filter_symbols_with_hybrid,
    pre_filter_symbols_with_voting,
)


def run_pre_filter(
    all_symbols: Optional[List[str]],
    primary_timeframe: str,
    limit: int,
    pre_filter_mode: str,
    fast_mode: bool,
    spc_preset: Optional[str],
    spc_volatility_adjustment: bool,
    spc_use_correlation_weights: bool,
    spc_time_decay_factor: Optional[float],
    spc_interpolation_mode: Optional[str],
    spc_min_flip_duration: Optional[int],
    spc_flip_confidence_threshold: Optional[float],
    spc_enable_mtf: bool,
    spc_mtf_timeframes: Optional[List[str]],
    spc_mtf_require_alignment: Optional[bool],
) -> Optional[List[str]]:
    """
    Run pre-filter on symbols.

    Args:
        all_symbols: List of all symbols to filter (or None to fetch from exchange)
        primary_timeframe: Primary timeframe for pre-filtering
        limit: Number of candles per symbol
        pre_filter_mode: Pre-filter mode ("voting" or "hybrid")
        fast_mode: Whether to use fast mode
        spc_preset: SPC preset (or None for custom)
        spc_volatility_adjustment: Enable volatility adjustment
        spc_use_correlation_weights: Enable correlation weights
        spc_time_decay_factor: Time decay factor
        spc_interpolation_mode: Interpolation mode
        spc_min_flip_duration: Minimum flip duration
        spc_flip_confidence_threshold: Flip confidence threshold
        spc_enable_mtf: Enable multi-timeframe
        spc_mtf_timeframes: Multi-timeframe timeframes
        spc_mtf_require_alignment: Require MTF alignment

    Returns:
        Filtered list of symbols, or None if pre-filter failed
    """
    try:
        log_info("=" * 60)
        log_info("PRE-FILTERING SYMBOLS")
        log_info("=" * 60)

        log_info("Step 1: Getting all symbols from exchange...")
        if all_symbols is None:
            exchange_manager = ExchangeManager()
            data_fetcher = DataFetcher(exchange_manager)
            all_symbols = data_fetcher.get_spot_symbols(exchange_name="binance", quote_currency="USDT")

        if not all_symbols:
            log_warn("No symbols found from exchange, skipping pre-filter")
            return None

        # Run pre-filter according to the selected mode
        if pre_filter_mode == "hybrid":
            try:
                pre_filtered_symbols = pre_filter_symbols_with_hybrid(
                    all_symbols=all_symbols,
                    timeframe=primary_timeframe,
                    limit=limit,
                    fast_mode=fast_mode,
                    spc_preset=spc_preset,
                    spc_volatility_adjustment=spc_volatility_adjustment,
                    spc_use_correlation_weights=spc_use_correlation_weights,
                    spc_time_decay_factor=spc_time_decay_factor,
                    spc_interpolation_mode=spc_interpolation_mode,
                    spc_min_flip_duration=spc_min_flip_duration,
                    spc_flip_confidence_threshold=spc_flip_confidence_threshold,
                    spc_enable_mtf=spc_enable_mtf,
                    spc_mtf_timeframes=spc_mtf_timeframes,
                    spc_mtf_require_alignment=spc_mtf_require_alignment,
                )
            except Exception as e:
                log_error(f"Exception during hybrid pre-filtering: {e}")
                return None
        else:
            try:
                pre_filtered_symbols = pre_filter_symbols_with_voting(
                    all_symbols=all_symbols,
                    timeframe=primary_timeframe,
                    limit=limit,
                    fast_mode=fast_mode,
                    spc_preset=spc_preset,
                    spc_volatility_adjustment=spc_volatility_adjustment,
                    spc_use_correlation_weights=spc_use_correlation_weights,
                    spc_time_decay_factor=spc_time_decay_factor,
                    spc_interpolation_mode=spc_interpolation_mode,
                    spc_min_flip_duration=spc_min_flip_duration,
                    spc_flip_confidence_threshold=spc_flip_confidence_threshold,
                    spc_enable_mtf=spc_enable_mtf,
                    spc_mtf_timeframes=spc_mtf_timeframes,
                    spc_mtf_require_alignment=spc_mtf_require_alignment,
                )
            except Exception as e:
                log_error(f"Exception during voting pre-filtering: {e}")
                return None

        if pre_filtered_symbols is not None:
            if len(pre_filtered_symbols) < len(all_symbols):
                msg = (
                    f"Pre-filtered: {len(all_symbols)} → {len(pre_filtered_symbols)} symbols (all symbols with signals)"
                )
                log_info(msg)
            elif len(pre_filtered_symbols) == len(all_symbols):
                log_info(f"Pre-filtered: All {len(all_symbols)} symbols have signals (no filtering applied)")
            else:
                log_warn("Pre-filtered symbols count is greater than all symbols. There may be an unexpected behavior.")
        else:
            log_info(f"Pre-filtered: No symbols with signals found, using all {len(all_symbols)} symbols")
            return None

        return pre_filtered_symbols

    except SymbolFetchError as e:
        log_error(f"Failed to fetch symbols from exchange: {e}")
        if e.is_retryable:
            log_error("This was a retryable error (network/rate limit). Please check your connection and try again.")
        else:
            log_error("This was a non-retryable error. Please check exchange configuration and API access.")
        log_warn("Continuing without pre-filter...")
        return None
    except Exception as e:
        log_error(f"Error during pre-filtering: {e}")
        log_warn("Continuing without pre-filter...")
        return None
