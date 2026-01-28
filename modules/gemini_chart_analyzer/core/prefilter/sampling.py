"""Stage 0 sampling helpers for prefilter workflow."""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from modules.common.ui.logging import log_info, log_success


def run_sampling_stage(
    *,
    all_symbols: List[str],
    stage0_sample_percentage: Optional[float],
    stage0_sampling_strategy: str,
    stage0_stratified_strata_count: int,
    stage0_hybrid_top_percentage: float,
) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """Run optional Stage 0 sampling and return symbols plus OHLCV cache."""
    data_cache: Dict[str, pd.DataFrame] = {}

    if stage0_sample_percentage is not None and 0 < stage0_sample_percentage < 100.0:
        from modules.gemini_chart_analyzer.core.prefilter.sampling_strategies import (
            SamplingStrategy,
            apply_sampling_strategy,
            get_symbol_volumes,
        )

        total_symbols = len(all_symbols)
        sample_count = max(1, int(total_symbols * stage0_sample_percentage / 100.0))

        log_info(
            f"[Pre-filter Stage 0] Sampling {sample_count} symbols "
            f"({stage0_sample_percentage}%) from {total_symbols} using '{stage0_sampling_strategy}' strategy..."
        )

        volumes = None
        strategy_enum = SamplingStrategy(stage0_sampling_strategy)

        # Initialize data fetcher if needed for volume or liquidity weighted
        temp_data_fetcher = None

        volume_required_strategies = {
            SamplingStrategy.VOLUME_WEIGHTED,
            SamplingStrategy.STRATIFIED,
            SamplingStrategy.TOP_N_HYBRID,
            SamplingStrategy.SYSTEMATIC,
            SamplingStrategy.LIQUIDITY_WEIGHTED,
        }

        if strategy_enum in volume_required_strategies:
            from modules.common.core.data_fetcher import DataFetcher
            from modules.common.core.exchange_manager import ExchangeManager

            if temp_data_fetcher is None:
                temp_exchange_manager = ExchangeManager()
                temp_data_fetcher = DataFetcher(temp_exchange_manager)

            log_info("[Pre-filter Stage 0] Fetching volume data for sampling...")
            volumes = get_symbol_volumes(all_symbols, temp_data_fetcher)

        # Pre-load OHLCV data for Liquidity Weighted strategy
        if strategy_enum == SamplingStrategy.LIQUIDITY_WEIGHTED:
            log_info(
                f"[Pre-filter Stage 0] Pre-loading OHLCV data for {len(all_symbols)} symbols (Liquidity Weighted)..."
            )

            if temp_data_fetcher is None:
                from modules.common.core.data_fetcher import DataFetcher
                from modules.common.core.exchange_manager import ExchangeManager

                temp_exchange_manager = ExchangeManager()
                temp_data_fetcher = DataFetcher(temp_exchange_manager)

            from concurrent.futures import ThreadPoolExecutor, as_completed

            metrics_lookback = 14
            fetch_limit = metrics_lookback * 2

            def fetch_symbol_data(symbol: str):
                try:
                    df, _ = temp_data_fetcher.fetch_ohlcv_with_fallback_exchange(
                        symbol, timeframe="1d", limit=fetch_limit
                    )
                    if df is not None and len(df) >= metrics_lookback:
                        return symbol, df
                    return symbol, None
                except Exception:
                    return symbol, None

            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_symbol = {executor.submit(fetch_symbol_data, sym): sym for sym in all_symbols}
                processed = 0
                for future in as_completed(future_to_symbol):
                    sym, df = future.result()
                    if df is not None:
                        data_cache[sym] = df
                    processed += 1
                    if processed % 100 == 0:
                        log_info(f"[Pre-filter Stage 0] Cached data for {processed}/{len(all_symbols)} symbols...")

            log_success(f"[Pre-filter Stage 0] Successfully cached OHLCV data for {len(data_cache)} symbols")

        sampled_symbols = apply_sampling_strategy(
            symbols=all_symbols,
            sample_percentage=stage0_sample_percentage,
            strategy=strategy_enum,
            volumes=volumes,
            strata_count=stage0_stratified_strata_count,
            top_percentage=stage0_hybrid_top_percentage,
            data_fetcher=temp_data_fetcher,
            ohlcv_cache=data_cache,
        )

        log_success(f"[Pre-filter Stage 0] Sampled {len(sampled_symbols)} symbols for processing")
        return sampled_symbols, data_cache

    return all_symbols, data_cache
