"""Helper module to integrate Gemini Chart Scanner sampling strategies into Auto Trade GUI."""

from typing import Dict, List

from modules.gemini_chart_analyzer.core.prefilter.sampling import (
    SamplingStrategy,
    apply_sampling_strategy,
    get_symbol_volumes,
)


def sample_symbols(
    all_symbols: List[str],
    sample_percentage: float,
    strategy: str,
    data_fetcher=None,
    **kwargs,
) -> List[str]:
    """
    Sample symbols using Gemini Chart Scanner's sampling strategies.

    Args:
        all_symbols: List of all available symbols
        sample_percentage: Percentage to sample (0-100)
        strategy: Sampling strategy name (e.g., 'random', 'stratified', 'volume_weighted')
        data_fetcher: Optional DataFetcher instance for volume data
        **kwargs: Additional strategy-specific parameters

    Returns:
        List of sampled symbols

    Available strategies:
        - random: Pure random sampling
        - volume_weighted: Higher volume symbols have higher probability
        - stratified: Divide into volume tiers, sample evenly from each (RECOMMENDED)
        - top_n_hybrid: Top N% by volume + random for rest
        - systematic: Every n-th symbol from volume-sorted list
        - liquidity_weighted: Combines volume/volatility/spread (advanced)
    """
    if sample_percentage <= 0 or sample_percentage >= 100:
        return all_symbols

    strategy_enum = SamplingStrategy(strategy)

    # Get volume data if needed
    volumes = None
    volume_required_strategies = {
        SamplingStrategy.VOLUME_WEIGHTED,
        SamplingStrategy.STRATIFIED,
        SamplingStrategy.TOP_N_HYBRID,
        SamplingStrategy.SYSTEMATIC,
        SamplingStrategy.LIQUIDITY_WEIGHTED,
    }

    if strategy_enum in volume_required_strategies and data_fetcher:
        volumes = get_symbol_volumes(all_symbols, data_fetcher)

    # Apply sampling strategy
    sampled = apply_sampling_strategy(
        symbols=all_symbols,
        sample_percentage=sample_percentage,
        strategy=strategy_enum,
        volumes=volumes,
        **kwargs,
    )

    return sampled


__all__ = ["sample_symbols", "SamplingStrategy"]
