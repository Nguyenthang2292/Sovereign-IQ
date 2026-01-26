"""
Symbol Sampling Strategies for Pre-filter Stage 0.

This module provides various sampling strategies to select a subset of symbols
before running the full pre-filter pipeline. Different strategies optimize for
different goals: speed, quality, diversity, or balance.
"""

import random
from enum import Enum
from typing import Dict, List, Optional

from modules.common.ui.logging import log_info, log_success


class SamplingStrategy(str, Enum):
    """Available sampling strategies for Stage 0."""

    RANDOM = "random"
    VOLUME_WEIGHTED = "volume_weighted"
    STRATIFIED = "stratified"
    TOP_N_HYBRID = "top_n_hybrid"
    SYSTEMATIC = "systematic"
    LIQUIDITY_WEIGHTED = "liquidity_weighted"


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
    sample_count = max(1, int(len(symbols) * sample_percentage / 100.0))
    sampled = random.sample(symbols, sample_count)
    log_info(f"[Random Sampling] Selected {len(sampled)} symbols uniformly at random")
    return sampled


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


def liquidity_weighted_sampling(
    symbols: List[str],
    sample_percentage: float,
    volumes: Dict[str, float],
    volatility_data: Optional[Dict[str, float]] = None,
    spread_data: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Liquidity-weighted sampling - combines volume, volatility, and spread.

    More sophisticated than pure volume weighting. Can incorporate spread/volatility
    for better liquidity assessment.

    Args:
        symbols: List of all symbols
        sample_percentage: Percentage to sample (0-100)
        volumes: Dictionary mapping symbol to volume
        volatility_data: Optional volatility data (not implemented yet)
        spread_data: Optional spread data (not implemented yet)

    Returns:
        List of liquidity-weighted sampled symbols
    """
    # For now, use volume as primary liquidity metric
    # TODO: Incorporate volatility and spread when available
    if volatility_data or spread_data:
        log_info("[Liquidity-Weighted Sampling] Volatility/spread data provided but not yet implemented")

    # Fall back to volume-weighted for now
    return volume_weighted_sampling(symbols, sample_percentage, volumes)


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
        return liquidity_weighted_sampling(symbols, sample_percentage, volumes, volatility_data, spread_data)
    else:
        log_info(f"[Sampling] Unknown strategy '{strategy}', falling back to random")
        return random_sampling(symbols, sample_percentage, volumes)


def get_symbol_volumes(symbols: List[str], data_fetcher) -> Dict[str, float]:
    """
    Get volume data for symbols from exchange.

    Uses the same approach as list_binance_futures_symbols to extract volume.

    Args:
        symbols: List of symbols to get volumes for
        data_fetcher: DataFetcher instance

    Returns:
        Dictionary mapping symbol to volume (quote volume)
    """
    volumes = {}

    try:
        # Use public API - load_markets() doesn't require authentication
        exchange = data_fetcher.exchange_manager.public.connect_to_exchange_with_no_credentials("binance")
    except Exception as exc:
        from modules.common.ui.logging import log_error

        log_error(f"Unable to connect to Binance for volume data: {exc}")
        return volumes

    try:
        # load_markets() is a public API call, no authentication needed
        markets = data_fetcher.exchange_manager.public.throttled_call(exchange.load_markets)
    except Exception as exc:
        from modules.common.ui.logging import log_error

        log_error(f"Failed to load Binance markets for volume data: {exc}")
        return volumes

    # Build symbol set for faster lookup
    symbol_set = set(symbols)

    for market in markets.values():
        symbol = data_fetcher.exchange_manager.normalize_symbol(market.get("symbol", ""))

        if symbol not in symbol_set:
            continue

        info = market.get("info", {})
        volume_str = info.get("volume") or info.get("quoteVolume") or info.get("turnover")
        try:
            volume = float(volume_str)
        except (TypeError, ValueError):
            volume = 0.0

        volumes[symbol] = volume

    log_success(f"Retrieved volume data for {len(volumes)}/{len(symbols)} symbols")
    return volumes
