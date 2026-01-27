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


def _calculate_volatility_and_spread(
    symbols: List[str],
    data_fetcher,
    timeframe: str = "1d",
    lookback: int = 14,
    use_rust: bool = True,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate volatility (ATR) and spread metrics for symbols.

    Volatility is measured as Average True Range (ATR) normalized by price.
    Spread is measured as (high - low) / close percentage.

    Args:
        symbols: List of symbols to calculate metrics for
        data_fetcher: DataFetcher instance for fetching OHLCV data
        timeframe: Timeframe for data (default: "1d" for daily)
        lookback: Lookback period for ATR calculation (default: 14)
        use_rust: Whether to use Rust backend for faster computation (default: True)

    Returns:
        Tuple of (volatility_dict, spread_dict) mapping symbol to metric value
    """
    import numpy as np
    from modules.common.ui.logging import log_error

    volatility = {}
    spread = {}

    # Try Rust implementation first if enabled
    if use_rust:
        try:
            import atc_rust

            # Batch fetch all OHLCV data
            ohlcv_data = {}
            for symbol in symbols:
                try:
                    df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                        symbol, timeframe=timeframe, limit=lookback * 2
                    )
                    if df is not None and len(df) >= lookback:
                        ohlcv_data[symbol] = {
                            "high": df["high"].values.astype(np.float64),
                            "low": df["low"].values.astype(np.float64),
                            "close": df["close"].values.astype(np.float64),
                        }
                except Exception:
                    continue

            if ohlcv_data:
                # Call Rust batch function for volatility/spread calculation
                results = atc_rust.compute_liquidity_metrics_batch(ohlcv_data, lookback)
                volatility = results.get("volatility", {})
                spread = results.get("spread", {})

                from modules.common.ui.logging import log_success

                log_success(
                    f"[Rust] Calculated volatility/spread for {len(volatility)} symbols (ATR lookback: {lookback})"
                )
                return volatility, spread
        except ImportError:
            from modules.common.ui.logging import log_warn

            log_warn("[Liquidity Metrics] Rust backend not available, using Python implementation")
        except Exception as e:
            log_error(f"[Liquidity Metrics] Rust calculation failed: {e}, falling back to Python")

    # Python fallback implementation
    from modules.common.ui.logging import log_info

    log_info(f"[Python] Calculating volatility/spread for {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            # Fetch OHLCV data
            df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol, timeframe=timeframe, limit=lookback * 2
            )
            if df is None or len(df) < lookback:
                continue

            # Calculate True Range: max(high-low, abs(high-close_prev), abs(low-close_prev))
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values

            # Shift close by 1 for previous close
            close_prev = np.roll(close, 1)
            close_prev[0] = np.nan

            tr = np.maximum(
                high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev))
            )

            # Calculate ATR (Average True Range) - simple moving average of TR
            # Skip first NaN value from rolled array
            atr = np.nanmean(tr[-lookback:])

            # Normalize ATR by current price (ATR%)
            current_price = close[-1]
            volatility[symbol] = (atr / current_price * 100.0) if current_price > 0 else 0.0

            # Calculate average spread percentage
            spread_pct = ((high - low) / close * 100.0)[-lookback:]
            spread[symbol] = float(np.nanmean(spread_pct))

        except Exception as e:
            log_error(f"[Liquidity Metrics] Error calculating metrics for {symbol}: {e}")
            continue

    from modules.common.ui.logging import log_success

    log_success(
        f"[Python] Calculated volatility/spread for {len(volatility)} symbols (ATR lookback: {lookback})"
    )
    return volatility, spread


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
    import numpy as np

    # If no volatility/spread data provided and calculate_metrics is enabled
    if calculate_metrics and (volatility_data is None or spread_data is None):
        if data_fetcher is None:
            log_info(
                "[Liquidity-Weighted Sampling] No data_fetcher provided, cannot calculate volatility/spread. "
                "Falling back to volume-weighted sampling."
            )
            return volume_weighted_sampling(symbols, sample_percentage, volumes)

        log_info("[Liquidity-Weighted Sampling] Calculating volatility and spread metrics...")
        calc_volatility, calc_spread = _calculate_volatility_and_spread(
            symbols, data_fetcher, timeframe="1d", lookback=14, use_rust=use_rust
        )

        # Use calculated data if original was None
        if volatility_data is None:
            volatility_data = calc_volatility
        if spread_data is None:
            spread_data = calc_spread

    # If still no data available, fall back to volume-weighted
    if not volatility_data and not spread_data:
        log_info("[Liquidity-Weighted Sampling] No volatility/spread data available, falling back to volume-weighted")
        return volume_weighted_sampling(symbols, sample_percentage, volumes)

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
        return liquidity_weighted_sampling(
            symbols, sample_percentage, volumes, volatility_data, spread_data, data_fetcher, use_rust=use_rust
        )
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
