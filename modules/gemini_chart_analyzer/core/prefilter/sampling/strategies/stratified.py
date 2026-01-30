"""Stratified sampling strategy."""

from typing import Dict, List

from modules.common.ui.logging import log_info
from modules.gemini_chart_analyzer.core.prefilter.sampling.strategies.random import random_sampling


def stratified_sampling(
    symbols: List[str],
    sample_percentage: float,
    volumes: Dict[str, float],
    strata_count: int = 3,
) -> List[str]:
    """
    Stratified sampling - divide symbols into volume tiers and sample evenly from each.

    Ensures representation across all liquidity levels (top/mid/low volume).
    Recommended for balanced market coverage.

    Args:
        symbols: List of all symbols
        sample_percentage: Percentage to sample (0-100)
        volumes: Dictionary mapping symbol to volume
        strata_count: Number of strata (default: 3 for top/mid/low)

    Returns:
        List of stratified sampled symbols
    """
    import random

    sample_count = max(1, int(len(symbols) * sample_percentage / 100.0))

    # Filter symbols that have volume data and sort by volume
    symbols_with_volume = [(s, volumes.get(s, 0.0)) for s in symbols]
    symbols_with_volume.sort(key=lambda x: x[1], reverse=True)

    if not symbols_with_volume:
        log_info("[Stratified Sampling] No volume data available, falling back to random")
        return random_sampling(symbols, sample_percentage, volumes)

    # Divide into strata
    strata_size = len(symbols_with_volume) // strata_count
    strata = []
    for i in range(strata_count):
        start_idx = i * strata_size
        end_idx = start_idx + strata_size if i < strata_count - 1 else len(symbols_with_volume)
        strata.append([s for s, _ in symbols_with_volume[start_idx:end_idx]])

    # Sample evenly from each stratum
    samples_per_stratum = sample_count // strata_count
    remainder = sample_count % strata_count

    sampled = []
    for i, stratum in enumerate(strata):
        # Distribute remainder across first strata
        stratum_sample_count = samples_per_stratum + (1 if i < remainder else 0)
        stratum_sample_count = min(stratum_sample_count, len(stratum))

        if stratum_sample_count > 0:
            sampled.extend(random.sample(stratum, stratum_sample_count))

    log_info(
        f"[Stratified Sampling] Selected {len(sampled)} symbols "
        f"from {strata_count} strata ({samples_per_stratum}±1 per stratum)"
    )
    return sampled
