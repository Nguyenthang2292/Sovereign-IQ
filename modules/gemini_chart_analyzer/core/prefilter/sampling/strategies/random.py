"""Random sampling strategy."""

from typing import Dict, List, Optional

from modules.common.ui.logging import log_info


def random_sampling(
    symbols: List[str],
    sample_percentage: float,
    volumes: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Pure random sampling - uniform probability for all symbols.

    Args:
        symbols: List of all symbols
        sample_percentage: Percentage to sample (0-100)
        volumes: Optional volume data (not used in this strategy)

    Returns:
        List of randomly sampled symbols
    """
    import random

    sample_count = max(1, int(len(symbols) * sample_percentage / 100.0))
    sampled = random.sample(symbols, sample_count)
    log_info(f"[Random Sampling] Selected {len(sampled)} symbols uniformly at random")
    return sampled
