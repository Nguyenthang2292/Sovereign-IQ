"""Volume-weighted sampling strategy."""

from typing import Dict, List

from modules.common.ui.logging import log_info
from modules.gemini_chart_analyzer.core.prefilter.sampling.strategies.random import random_sampling


def volume_weighted_sampling(
    symbols: List[str],
    sample_percentage: float,
    volumes: Dict[str, float],
) -> List[str]:
    """
    Volume-weighted sampling - higher volume symbols have higher probability.

    Prioritizes symbols with high volume → increases likelihood of good signals.
    Uses weighted random sampling based on volume.

    Args:
        symbols: List of all symbols
        sample_percentage: Percentage to sample (0-100)
        volumes: Dictionary mapping symbol to volume

    Returns:
        List of volume-weighted sampled symbols
    """
    import random

    sample_count = max(1, int(len(symbols) * sample_percentage / 100.0))

    # Filter symbols that have volume data
    symbols_with_volume = [(s, volumes.get(s, 0.0)) for s in symbols]
    symbols_with_volume = [(s, v) for s, v in symbols_with_volume if v > 0]

    if not symbols_with_volume:
        log_info("[Volume-Weighted Sampling] No volume data available, falling back to random")
        return random_sampling(symbols, sample_percentage, volumes)

    # Calculate total volume for normalization
    total_volume = sum(v for _, v in symbols_with_volume)
    if total_volume == 0:
        log_info("[Volume-Weighted Sampling] Total volume is zero, falling back to random")
        return random_sampling(symbols, sample_percentage, volumes)

    # Calculate weights (probabilities)
    weights = [v / total_volume for _, v in symbols_with_volume]
    symbol_list = [s for s, _ in symbols_with_volume]

    # Weighted random sampling
    sampled = random.choices(symbol_list, weights=weights, k=sample_count)

    # Remove duplicates while preserving order
    seen = set()
    unique_sampled = []
    for s in sampled:
        if s not in seen:
            seen.add(s)
            unique_sampled.append(s)

    log_info(
        f"[Volume-Weighted Sampling] Selected {len(unique_sampled)} symbols "
        f"(weighted by volume, avg volume: {sum(volumes.get(s, 0) for s in unique_sampled) / len(unique_sampled):.2f})"
    )
    return unique_sampled
