"""Liquidity-weighted sampling strategy."""

from typing import Dict, List, Optional

import numpy as np

import pandas as pd

from modules.common.ui.logging import log_info
from modules.gemini_chart_analyzer.core.prefilter.sampling.metrics import calculate_volatility_and_spread
from modules.gemini_chart_analyzer.core.prefilter.sampling.strategies.random import random_sampling


def liquidity_weighted_sampling(
    symbols: List[str],
    sample_percentage: float,
    volumes: Dict[str, float],
    volatility_data: Optional[Dict[str, float]] = None,
    spread_data: Optional[Dict[str, float]] = None,
    data_fetcher=None,
    calculate_metrics: bool = True,
    volatility_weight: float = 0.4,
    spread_weight: float = 0.2,
    volume_weight: float = 0.4,
    prefer_low_volatility: bool = False,
    use_rust: bool = True,
    ohlcv_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[str]:
    """
    Liquidity-weighted sampling - combines volume, volatility, and spread.

    More sophisticated than pure volume weighting. Incorporates spread/volatility
    for better liquidity assessment. High liquidity = high volume + low spread + moderate volatility.

    Liquidity Score Formula:
        score = volume_weight * norm(volume)
              + spread_weight * (1 - norm(spread))    [lower spread = better liquidity]
              + volatility_weight * volatility_term   [depends on prefer_low_volatility]

    Args:
        symbols: List of all symbols
        sample_percentage: Percentage to sample (0-100)
        volumes: Dictionary mapping symbol to volume
        volatility_data: Optional volatility data (ATR%). If None and calculate_metrics=True, will calculate
        spread_data: Optional spread data (%). If None and calculate_metrics=True, will calculate
        data_fetcher: DataFetcher instance (required if calculate_metrics=True and data not provided)
        calculate_metrics: Whether to calculate volatility/spread if not provided (default: True)
        volatility_weight: Weight for volatility in liquidity score (default: 0.4)
        spread_weight: Weight for spread in liquidity score (default: 0.2)
        volume_weight: Weight for volume in liquidity score (default: 0.4)
        prefer_low_volatility: If True, prefer low volatility (stable). If False, prefer moderate volatility
                              (trading opportunity). Default: False
        use_rust: Whether to use Rust backend for metric calculation (default: True)

    Returns:
        List of liquidity-weighted sampled symbols
    """
    import random

    # If no volatility/spread data provided and calculate_metrics is enabled
    if calculate_metrics and (volatility_data is None or spread_data is None):
        if data_fetcher is None:
            log_info(
                "[Liquidity-Weighted Sampling] No data_fetcher provided, cannot calculate volatility/spread. "
                "Falling back to volume-weighted sampling."
            )
            return random_sampling(symbols, sample_percentage, volumes)

        log_info("[Liquidity-Weighted Sampling] Calculating volatility and spread metrics...")
        calc_volatility, calc_spread = calculate_volatility_and_spread(
            symbols, data_fetcher, timeframe="1d", lookback=14, use_rust=use_rust, ohlcv_cache=ohlcv_cache
        )

        # Use calculated data if original was None
        if volatility_data is None:
            volatility_data = calc_volatility
        if spread_data is None:
            spread_data = calc_spread

    # If still no data available, fall back to volume-weighted
    if not volatility_data and not spread_data:
        log_info("[Liquidity-Weighted Sampling] No volatility/spread data available, falling back to volume-weighted")
        return random_sampling(symbols, sample_percentage, volumes)

    sample_count = max(1, int(len(symbols) * sample_percentage / 100.0))

    # Filter symbols with volume data
    symbols_with_data = []
    for s in symbols:
        vol = volumes.get(s, 0.0)
        if vol > 0:
            symbols_with_data.append(s)

    if not symbols_with_data:
        log_info("[Liquidity-Weighted Sampling] No symbols with volume data, falling back to random")
        return random_sampling(symbols, sample_percentage, volumes)

    # Calculate liquidity scores
    liquidity_scores = {}

    # Get all metric values for normalization
    vol_values = [volumes.get(s, 0.0) for s in symbols_with_data]
    volatility_values = [volatility_data.get(s, 0.0) for s in symbols_with_data] if volatility_data else []
    spread_values = [spread_data.get(s, 0.0) for s in symbols_with_data] if spread_data else []

    # Normalize to 0-1 range (min-max normalization)
    def normalize(values):
        if not values or len(values) == 0:
            return {}
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return {symbols_with_data[i]: 0.5 for i in range(len(values))}
        return {symbols_with_data[i]: (values[i] - min_val) / (max_val - min_val) for i in range(len(values))}

    norm_volume = normalize(vol_values)
    norm_volatility = normalize(volatility_values) if volatility_values else {}
    norm_spread = normalize(spread_values) if spread_values else {}

    # Calculate composite liquidity score
    for s in symbols_with_data:
        score = 0.0

        # Volume component (higher is better)
        score += volume_weight * norm_volume.get(s, 0.0)

        # Spread component (lower is better, so invert)
        if norm_spread:
            score += spread_weight * (1.0 - norm_spread.get(s, 0.0))

        # Volatility component (depends on preference)
        if norm_volatility:
            vol_norm = norm_volatility.get(s, 0.0)
            if prefer_low_volatility:
                # Prefer low volatility (stable assets)
                score += volatility_weight * (1.0 - vol_norm)
            else:
                # Prefer moderate volatility (trading opportunity)
                # Use inverted parabola: 1 - 4*(x - 0.5)^2, peaks at x=0.5
                score += volatility_weight * (1.0 - 4.0 * (vol_norm - 0.5) ** 2)

        liquidity_scores[s] = max(score, 0.001)  # Ensure positive scores

    # Weighted random sampling based on liquidity scores
    total_score = sum(liquidity_scores.values())
    if total_score == 0:
        log_info("[Liquidity-Weighted Sampling] All scores are zero, falling back to random")
        return random_sampling(symbols, sample_percentage, volumes)

    weights = [liquidity_scores[s] / total_score for s in symbols_with_data]

    # Perform weighted sampling
    sampled = random.choices(symbols_with_data, weights=weights, k=sample_count)

    # Remove duplicates while preserving order
    seen = set()
    unique_sampled = []
    for s in sampled:
        if s not in seen:
            seen.add(s)
            unique_sampled.append(s)

    # Calculate average metrics for sampled symbols
    avg_volume = np.mean([volumes.get(s, 0) for s in unique_sampled]) if unique_sampled else 0
    avg_volatility = (
        np.mean([volatility_data.get(s, 0) for s in unique_sampled]) if volatility_data and unique_sampled else 0
    )
    avg_spread = np.mean([spread_data.get(s, 0) for s in unique_sampled]) if spread_data and unique_sampled else 0

    log_info(
        f"[Liquidity-Weighted Sampling] Selected {len(unique_sampled)} symbols "
        f"(avg volume: {avg_volume:.2f}, avg volatility: {avg_volatility:.2f}%, avg spread: {avg_spread:.2f}%)"
    )

    return unique_sampled
