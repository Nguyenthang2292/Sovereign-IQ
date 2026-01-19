from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from modules.simplified_percentile_clustering.config.mean_reversion_config import (
    MeanReversionConfig,
)
from modules.simplified_percentile_clustering.core.clustering import (
    ClusteringResult,
    compute_clustering,
)
from modules.simplified_percentile_clustering.utils.helpers import (
    safe_isna,
    vectorized_extreme_duration,
)

"""
Mean Reversion Strategy.

This strategy generates signals when market is at cluster extremes
and expects mean reversion back to center cluster.

Strategy Logic:
--------------
1. LONG Signal:
    - Market in k0 cluster (lower extreme)
    - Real_clust near 0 (far from center)
    - Expecting reversion to k1 or k2
    - Price showing signs of reversal

2. SHORT Signal:
    - Market in k2 or k1 cluster (upper extreme)
    - Real_clust near maximum (far from center)
    - Expecting reversion to k0 or k1
    - Price showing signs of reversal

3. NEUTRAL Signal:
    - Market near center cluster
    - No extreme conditions
    - Ambiguous signals
"""


def _detect_reversal(close: pd.Series, i: int, lookback: int, direction: str) -> bool:
    """Detect price reversal signal."""
    if i < lookback:
        return False

    recent_prices = close.iloc[i - lookback : i + 1]
    if len(recent_prices) < 2:
        return False

    if direction == "bullish":
        # Check for bullish reversal: recent low followed by higher close
        recent_min_idx = recent_prices.idxmin()
        if recent_min_idx == recent_prices.index[-1]:
            return False
        min_idx_pos = recent_prices.index.get_loc(recent_min_idx)
        if min_idx_pos < len(recent_prices) - 1:
            # Price increased after the low
            return recent_prices.iloc[-1] > recent_prices.iloc[min_idx_pos]
        return False
    else:  # bearish
        # Check for bearish reversal: recent high followed by lower close
        recent_max_idx = recent_prices.idxmax()
        if recent_max_idx == recent_prices.index[-1]:
            return False
        max_idx_pos = recent_prices.index.get_loc(recent_max_idx)
        if max_idx_pos < len(recent_prices) - 1:
            # Price decreased after the high
            return recent_prices.iloc[-1] < recent_prices.iloc[max_idx_pos]
        return False


def generate_signals_mean_reversion(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    volume: Optional[pd.Series] = None,
    clustering_result: Optional[ClusteringResult] = None,
    config: Optional[MeanReversionConfig] = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Generate trading signals based on mean reversion.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series (optional, for future enhancements).
        clustering_result: Pre-computed clustering result (optional).
        config: Strategy configuration.

    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
        - metadata: DataFrame with additional signal metadata
    """
    if config is None:
        config = MeanReversionConfig()

    # Ensure targets are updated if clustering_config is set
    if config.clustering_config is not None:
        config.update_targets()

    # Compute clustering if not provided
    if clustering_result is None:
        clustering_result = compute_clustering(high, low, close, config=config.clustering_config)

    signals = pd.Series(0, index=close.index, dtype=int)
    signal_strength = pd.Series(0.0, index=close.index, dtype=float)

    # Compute RSI if confirmation is enabled
    rsi_vals = None
    if config.use_rsi_confirmation:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=config.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_vals = 100 - (100 / (1 + rs))
        rsi_vals = rsi_vals.fillna(50.0)

    # Determine max real_clust value based on k
    k = 2
    if config.clustering_config:
        k = config.clustering_config.k
    max_real_clust = 2.0 if k == 3 else 1.0

    # Track extreme duration using vectorized operations
    extreme_duration, in_extreme = vectorized_extreme_duration(
        clustering_result.real_clust,
        config.extreme_threshold,
        max_real_clust,
    )

    # Metadata columns
    metadata = {
        "cluster_val": clustering_result.cluster_val,
        "real_clust": clustering_result.real_clust,
        "extreme_duration": extreme_duration,
        "in_extreme": in_extreme,
        "price_change": close.pct_change(),
    }
    if rsi_vals is not None:
        metadata["rsi"] = rsi_vals

    for i in range(len(close)):
        real_clust = metadata["real_clust"].iloc[i]
        cluster_val = metadata["cluster_val"].iloc[i]
        duration = metadata["extreme_duration"].iloc[i]
        is_extreme = metadata["in_extreme"].iloc[i]

        if safe_isna(real_clust) or safe_isna(cluster_val):
            continue

        # Check if in extreme long enough
        if not is_extreme or duration < config.min_extreme_duration:
            continue

        # Bullish reversion signal (from lower extreme)
        is_lower_extreme = real_clust <= config.extreme_threshold
        if is_lower_extreme:
            reversal_confirmed = True
            if config.require_reversal_signal:
                reversal_confirmed = _detect_reversal(close, i, config.reversal_lookback, "bullish")

            # Check RSI confirmation if enabled
            rsi_confirmed = True
            if config.use_rsi_confirmation and rsi_vals is not None:
                rsi_val = rsi_vals.iloc[i]
                rsi_confirmed = rsi_val < config.rsi_oversold

            if reversal_confirmed and rsi_confirmed:
                # Calculate distance to target
                distance_to_target = abs(real_clust - config.bullish_reversion_target)
                max_distance = max_real_clust
                strength = 1.0 - min(distance_to_target / max_distance, 1.0)

                # Adjust strength based on how extreme (lower = stronger)
                extreme_strength = 1.0 - (real_clust / config.extreme_threshold)
                combined_strength = (strength + extreme_strength) / 2.0

                if combined_strength >= config.min_signal_strength:
                    signals.iloc[i] = 1  # LONG
                    signal_strength.iloc[i] = combined_strength

        # Bearish reversion signal (from upper extreme)
        is_upper_extreme = real_clust >= (max_real_clust - config.extreme_threshold)
        if is_upper_extreme:
            reversal_confirmed = True
            if config.require_reversal_signal:
                reversal_confirmed = _detect_reversal(close, i, config.reversal_lookback, "bearish")

            # Check RSI confirmation if enabled
            rsi_confirmed = True
            if config.use_rsi_confirmation and rsi_vals is not None:
                rsi_val = rsi_vals.iloc[i]
                rsi_confirmed = rsi_val > config.rsi_overbought

            if reversal_confirmed and rsi_confirmed:
                # Calculate distance to target
                distance_to_target = abs(real_clust - config.bearish_reversion_target)
                max_distance = max_real_clust
                strength = 1.0 - min(distance_to_target / max_distance, 1.0)

                # Adjust strength based on how extreme (higher = stronger)
                extreme_strength = (real_clust - (max_real_clust - config.extreme_threshold)) / config.extreme_threshold
                combined_strength = (strength + extreme_strength) / 2.0

                if combined_strength >= config.min_signal_strength:
                    signals.iloc[i] = -1  # SHORT
                    signal_strength.iloc[i] = combined_strength

    metadata_df = pd.DataFrame(metadata, index=close.index)
    metadata_df["signal"] = signals
    metadata_df["signal_strength"] = signal_strength

    return signals, signal_strength, metadata_df


__all__ = ["MeanReversionConfig", "generate_signals_mean_reversion"]
