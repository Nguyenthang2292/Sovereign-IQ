"""Factory for selecting and applying sampling strategies."""

from typing import Dict, List, Optional

from modules.common.ui.logging import log_info
from modules.gemini_chart_analyzer.core.prefilter.sampling.base import SamplingStrategy
from modules.gemini_chart_analyzer.core.prefilter.sampling.strategies import (
    liquidity_weighted_sampling,
    random_sampling,
    stratified_sampling,
    systematic_sampling,
    top_n_hybrid_sampling,
    volume_weighted_sampling,
)


def apply_sampling_strategy(
    symbols: List[str],
    sample_percentage: float,
    strategy: SamplingStrategy,
    volumes: Optional[Dict[str, float]] = None,
    **kwargs,
) -> List[str]:
    """
    Apply the specified sampling strategy to select symbols.

    Args:
        symbols: List of all symbols
        sample_percentage: Percentage to sample (0-100)
        strategy: Sampling strategy to use
        volumes: Optional volume data (required for non-random strategies)
        **kwargs: Additional strategy-specific parameters
            - strata_count: For stratified sampling (default: 3)
            - top_percentage: For top_n_hybrid sampling (default: 50.0)
            - volatility_data: For liquidity_weighted sampling (optional)
            - spread_data: For liquidity_weighted sampling (optional)
            - data_fetcher: For liquidity_weighted sampling (required if calculating metrics)
            - use_rust: For liquidity_weighted sampling (default: True)

    Returns:
        List of sampled symbols

    Raises:
        ValueError: If strategy requires volume data but none provided
    """
    if sample_percentage <= 0 or sample_percentage >= 100:
        log_info(f"[Sampling] Invalid percentage {sample_percentage}, returning all symbols")
        return symbols

    # Strategies that require volume data
    volume_required_strategies = {
        SamplingStrategy.VOLUME_WEIGHTED,
        SamplingStrategy.STRATIFIED,
        SamplingStrategy.TOP_N_HYBRID,
        SamplingStrategy.SYSTEMATIC,
        SamplingStrategy.LIQUIDITY_WEIGHTED,
    }

    if strategy in volume_required_strategies and not volumes:
        log_info(f"[Sampling] Strategy '{strategy}' requires volume data, falling back to random")
        strategy = SamplingStrategy.RANDOM

    # Apply strategy
    if strategy == SamplingStrategy.RANDOM:
        return random_sampling(symbols, sample_percentage, volumes)
    elif strategy == SamplingStrategy.VOLUME_WEIGHTED:
        return volume_weighted_sampling(symbols, sample_percentage, volumes)
    elif strategy == SamplingStrategy.STRATIFIED:
        strata_count = kwargs.get("strata_count", 3)
        return stratified_sampling(symbols, sample_percentage, volumes, strata_count)
    elif strategy == SamplingStrategy.TOP_N_HYBRID:
        top_percentage = kwargs.get("top_percentage", 50.0)
        return top_n_hybrid_sampling(symbols, sample_percentage, volumes, top_percentage)
    elif strategy == SamplingStrategy.SYSTEMATIC:
        return systematic_sampling(symbols, sample_percentage, volumes)
    elif strategy == SamplingStrategy.LIQUIDITY_WEIGHTED:
        volatility_data = kwargs.get("volatility_data")
        spread_data = kwargs.get("spread_data")
        data_fetcher = kwargs.get("data_fetcher")
        use_rust = kwargs.get("use_rust", True)
        ohlcv_cache = kwargs.get("ohlcv_cache")
        return liquidity_weighted_sampling(
            symbols,
            sample_percentage,
            volumes,
            volatility_data,
            spread_data,
            data_fetcher,
            use_rust=use_rust,
            ohlcv_cache=ohlcv_cache,
        )
    else:
        log_info(f"[Sampling] Unknown strategy '{strategy}', falling back to random")
        return random_sampling(symbols, sample_percentage, volumes)
