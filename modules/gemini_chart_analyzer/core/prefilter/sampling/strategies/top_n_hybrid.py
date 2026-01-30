"""Top-N hybrid sampling strategy."""

from typing import Dict, List

from modules.common.ui.logging import log_info
from modules.gemini_chart_analyzer.core.prefilter.sampling.strategies.random import random_sampling


def top_n_hybrid_sampling(
    symbols: List[str],
    sample_percentage: float,
    volumes: Dict[str, float],
    top_percentage: float = 50.0,
) -> List[str]:
    """
    Top-N + Random hybrid - take top N% by volume, rest random.

    Balances quality (high volume) with diversity (random selection).

    Args:
        symbols: List of all symbols
        sample_percentage: Percentage to sample (0-100)
        volumes: Dictionary mapping symbol to volume
        top_percentage: Percentage of sample to take from top volume (default: 50%)

    Returns:
        List of hybrid sampled symbols
    """
    import random

    sample_count = max(1, int(len(symbols) * sample_percentage / 100.0))

    # Filter symbols that have volume data and sort by volume
    symbols_with_volume = [(s, volumes.get(s, 0.0)) for s in symbols]
    symbols_with_volume.sort(key=lambda x: x[1], reverse=True)

    if not symbols_with_volume:
        log_info("[Top-N Hybrid Sampling] No volume data available, falling back to random")
        return random_sampling(symbols, sample_percentage, volumes)

    # Calculate split
    top_count = max(1, int(sample_count * top_percentage / 100.0))
    random_count = sample_count - top_count

    # Take top N by volume
    top_symbols = [s for s, _ in symbols_with_volume[:top_count]]

    # Take random from remaining
    remaining_symbols = [s for s, _ in symbols_with_volume[top_count:]]
    if random_count > 0 and remaining_symbols:
        random_symbols = random.sample(remaining_symbols, min(random_count, len(remaining_symbols)))
    else:
        random_symbols = []

    sampled = top_symbols + random_symbols

    log_info(
        f"[Top-N Hybrid Sampling] Selected {len(sampled)} symbols "
        f"({len(top_symbols)} top by volume + {len(random_symbols)} random)"
    )
    return sampled
