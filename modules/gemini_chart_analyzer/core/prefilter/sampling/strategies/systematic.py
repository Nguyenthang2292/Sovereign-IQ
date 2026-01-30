"""Systematic sampling strategy."""

from typing import Dict, List

from modules.common.ui.logging import log_info
from modules.gemini_chart_analyzer.core.prefilter.sampling.strategies.random import random_sampling


def systematic_sampling(
    symbols: List[str],
    sample_percentage: float,
    volumes: Dict[str, float],
) -> List[str]:
    """
    Systematic sampling - take every n-th symbol from volume-sorted list.

    Simple and ensures even distribution across volume spectrum.

    Args:
        symbols: List of all symbols
        sample_percentage: Percentage to sample (0-100)
        volumes: Dictionary mapping symbol to volume

    Returns:
        List of systematically sampled symbols
    """
    sample_count = max(1, int(len(symbols) * sample_percentage / 100.0))

    # Filter symbols that have volume data and sort by volume
    symbols_with_volume = [(s, volumes.get(s, 0.0)) for s in symbols]
    symbols_with_volume.sort(key=lambda x: x[1], reverse=True)

    if not symbols_with_volume:
        log_info("[Systematic Sampling] No volume data available, falling back to random")
        return random_sampling(symbols, sample_percentage, volumes)

    # Calculate step size
    step = max(1, len(symbols_with_volume) // sample_count)

    # Take every n-th symbol
    sampled = [s for i, (s, _) in enumerate(symbols_with_volume) if i % step == 0][:sample_count]

    log_info(f"[Systematic Sampling] Selected {len(sampled)} symbols (every {step}-th symbol)")
    return sampled
