
from typing import Optional, Tuple

import pandas as pd

from __future__ import annotations
from modules.simplified_percentile_clustering.config.cluster_transition_config import (
from __future__ import annotations
from modules.simplified_percentile_clustering.config.cluster_transition_config import (

"""
Cluster Transition Strategy.

This strategy generates trading signals based on cluster transitions.
When the market transitions from one cluster to another, it may indicate
a regime change and potential trading opportunity.

Strategy Logic:
--------------
1. LONG Signal:
   - Transition from k0 (lower cluster) to k1 or k2 (higher clusters)
   - Real_clust value increasing and crossing cluster boundaries
   - Confirmation: price moving in same direction

2. SHORT Signal:
   - Transition from k2 or k1 (higher clusters) to k0 (lower cluster)
   - Real_clust value decreasing and crossing cluster boundaries
   - Confirmation: price moving in same direction

3. NEUTRAL Signal:
   - No cluster transition
   - Real_clust staying within same cluster
   - Ambiguous transitions (rel_pos near 0.5)
"""




    ClusterTransitionConfig,
)
from modules.simplified_percentile_clustering.core.clustering import (
    ClusteringResult,
    compute_clustering,
)
from modules.simplified_percentile_clustering.utils.helpers import (
    safe_isna,
    vectorized_transition_detection,
)


def generate_signals_cluster_transition(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    clustering_result: Optional[ClusteringResult] = None,
    config: Optional[ClusterTransitionConfig] = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Generate trading signals based on cluster transitions.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        clustering_result: Pre-computed clustering result (optional).
        config: Strategy configuration.

    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
        - metadata: DataFrame with additional signal metadata
    """
    if config is None:
        config = ClusterTransitionConfig()

    # Compute clustering if not provided
    if clustering_result is None:
        clustering_result = compute_clustering(high, low, close, config=config.clustering_config)

    signals = pd.Series(0, index=close.index, dtype=int)
    signal_strength = pd.Series(0.0, index=close.index, dtype=float)

    # Pre-compute metadata using vectorized operations
    prev_cluster_val = clustering_result.cluster_val.shift(1)
    prev_real_clust = clustering_result.real_clust.shift(1)
    price_change = close.pct_change()

    # Vectorized transition detection
    bullish_mask, bearish_mask = vectorized_transition_detection(
        prev_cluster_val,
        clustering_result.cluster_val,
        config.bullish_transitions,
        config.bearish_transitions,
    )

    # Calculate signal strength components using vectorized operations
    real_clust_change = (clustering_result.real_clust - prev_real_clust).abs()
    max_possible_change = 2.0 if config.clustering_config and config.clustering_config.k == 3 else 1.0
    strength_from_movement = (real_clust_change / max_possible_change).clip(upper=1.0)
    rel_pos_strength = 1.0 - clustering_result.rel_pos.clip(upper=1.0)
    combined_strength = (strength_from_movement + rel_pos_strength) / 2.0

    # Apply price confirmation if required
    if config.require_price_confirmation:
        # Only allow signals when price_change is not NaN and matches direction
        bullish_mask &= (price_change > 0) & price_change.notna()
        bearish_mask &= (price_change < 0) & price_change.notna()

    # Apply minimum signal strength filter
    strength_mask = combined_strength >= config.min_signal_strength
    bullish_mask &= strength_mask
    bearish_mask &= strength_mask

    # Set signals
    signals.loc[bullish_mask] = 1
    signals.loc[bearish_mask] = -1
    signal_strength.loc[bullish_mask] = combined_strength.loc[bullish_mask]
    signal_strength.loc[bearish_mask] = combined_strength.loc[bearish_mask]

    # Handle real_clust crossing if enabled (still need loop for complex logic)
    if config.use_real_clust_cross:
        for i in range(1, len(close)):
            prev_real = prev_real_clust.iloc[i]
            curr_real = clustering_result.real_clust.iloc[i]

            # Skip if missing data
            if safe_isna(prev_real) or safe_isna(curr_real):
                continue

            # Only process if no transition signal already set
            if signals.iloc[i] != 0:
                continue

            curr_strength = combined_strength.iloc[i] if not safe_isna(combined_strength.iloc[i]) else 0.0

            # Crossing from below k0.5 to above (bullish)
            if prev_real < 0.5 and curr_real >= 0.5:
                if config.clustering_config and config.clustering_config.k == 3:
                    if prev_real < 1.5 and curr_real >= 1.5:
                        # Crossing to k2
                        if curr_strength >= config.min_signal_strength:
                            signals.iloc[i] = 1
                            signal_strength.iloc[i] = curr_strength * 0.8
                else:
                    # Crossing to k1
                    if curr_strength >= config.min_signal_strength:
                        signals.iloc[i] = 1
                        signal_strength.iloc[i] = curr_strength * 0.8

            # Crossing from above k0.5 to below (bearish)
            elif prev_real > 0.5 and curr_real <= 0.5:
                if config.clustering_config and config.clustering_config.k == 3:
                    if prev_real > 1.5 and curr_real <= 1.5:
                        # Crossing from k2
                        if curr_strength >= config.min_signal_strength:
                            signals.iloc[i] = -1
                            signal_strength.iloc[i] = curr_strength * 0.8
                else:
                    # Crossing to k0
                    if curr_strength >= config.min_signal_strength:
                        signals.iloc[i] = -1
                        signal_strength.iloc[i] = curr_strength * 0.8

    # Build metadata DataFrame
    metadata = {
        "cluster_val": clustering_result.cluster_val,
        "prev_cluster_val": prev_cluster_val,
        "real_clust": clustering_result.real_clust,
        "prev_real_clust": prev_real_clust,
        "rel_pos": clustering_result.rel_pos,
        "price_change": price_change,
    }
    metadata_df = pd.DataFrame(metadata, index=close.index)
    metadata_df["signal"] = signals
    metadata_df["signal_strength"] = signal_strength

    return signals, signal_strength, metadata_df


__all__ = ["ClusterTransitionConfig", "generate_signals_cluster_transition"]
