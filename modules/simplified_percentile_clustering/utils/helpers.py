"""
Helper functions for Simplified Percentile Clustering.

Provides optimized utility functions for common operations.
"""

from typing import Tuple
import warnings
import numpy as np
import pandas as pd


def safe_isna(value) -> bool:
    """
    Safe NaN check that works with both pandas and numpy types.
    
    Uses pd.isna() which is optimized for pandas Series and handles
    both pandas and numpy types efficiently.
    
    Args:
        value: Value to check (can be scalar, Series, or array)
        
    Returns:
        bool or Series/array: True where value is NaN
    """
    return pd.isna(value)


def safe_isfinite(value) -> bool:
    """
    Safe finite check that works with both pandas and numpy types.
    
    Args:
        value: Value to check (can be scalar, Series, or array)
        
    Returns:
        bool or Series/array: True where value is finite
    """
    if isinstance(value, pd.Series):
        return pd.Series(np.isfinite(value.values), index=value.index)
    return np.isfinite(value)


def vectorized_min_distance(
    feature_val: pd.Series,
    centers: pd.DataFrame,
) -> pd.Series:
    """
    Compute minimum distance from feature values to centers using vectorized operations.
    
    This is a vectorized replacement for the loop-based _compute_distance_single method.
    
    Args:
        feature_val: Feature values as Series
        centers: DataFrame with center columns (k0, k1, k2)
        
    Returns:
        Series with minimum distances for each timestamp
    """
    if len(feature_val) == 0 or len(centers) == 0:
        return pd.Series(dtype=float, index=feature_val.index)
    
    # Align indices
    if not feature_val.index.equals(centers.index):
        # Reindex to common index
        common_index = feature_val.index.intersection(centers.index)
        feature_val = feature_val.reindex(common_index)
        centers = centers.reindex(common_index)
    
    # Convert to numpy arrays for vectorized operations
    feature_arr = feature_val.values[:, None]  # Shape: (n, 1)
    centers_arr = centers.values  # Shape: (n, k)
    
    # Compute distances: abs(feature - center) for all centers at once
    # Broadcasting: (n, 1) - (n, k) -> (n, k)
    distances = np.abs(feature_arr - centers_arr)
    
    # Find minimum distance along center axis (axis=1)
    # Use nanmin to ignore NaN values
    # Suppress warning for All-NaN slices (expected behavior when all values are NaN)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
        min_dist = np.nanmin(distances, axis=1)
    
    # Convert back to Series
    return pd.Series(min_dist, index=feature_val.index)


def vectorized_min_and_second_min(
    dist_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find minimum and second minimum distances using vectorized operations.
    
    This replaces the loop-based approach in the compute() method.
    
    Args:
        dist_array: Array of shape (n, k) with distances to k centers for n timestamps
        
    Returns:
        Tuple of:
        - min_dist: Array of minimum distances (n,)
        - second_min_dist: Array of second minimum distances (n,)
        - cluster_val: Array of cluster indices with minimum distance (n,)
        - second_cluster_val: Array of cluster indices with second minimum distance (n,)
    """
    n, k = dist_array.shape
    
    # Initialize result arrays
    min_dist = np.full(n, np.nan)
    second_min_dist = np.full(n, np.nan)
    cluster_val = np.full(n, np.nan)
    second_cluster_val = np.full(n, np.nan)
    
    # Create mask for valid (non-NaN) values
    valid_mask = ~np.isnan(dist_array)
    
    # Process each row
    for i in range(n):
        row = dist_array[i, :]
        valid = valid_mask[i, :]
        
        if not np.any(valid):
            continue
        
        valid_distances = row[valid]
        valid_indices = np.where(valid)[0]
        
        if len(valid_distances) == 0:
            continue
        
        # Sort valid distances to find min and second min
        sorted_idx = np.argsort(valid_distances)
        
        # Minimum
        min_idx = valid_indices[sorted_idx[0]]
        min_dist[i] = valid_distances[sorted_idx[0]]
        cluster_val[i] = min_idx
        
        # Second minimum (if exists)
        if len(valid_distances) > 1:
            second_min_idx = valid_indices[sorted_idx[1]]
            second_min_dist[i] = valid_distances[sorted_idx[1]]
            second_cluster_val[i] = second_min_idx
    
    return min_dist, second_min_dist, cluster_val, second_cluster_val


def normalize_cluster_name(cluster_val: float) -> str:
    """
    Normalize cluster value to cluster name string.
    
    Args:
        cluster_val: Cluster value (0, 1, or 2)
        
    Returns:
        Cluster name string ("k0", "k1", or "k2")
    """
    if pd.isna(cluster_val):
        return None
    
    cluster_int = int(cluster_val)
    if cluster_int in [0, 1, 2]:
        return f"k{cluster_int}"
    
    return None


def vectorized_cluster_duration(cluster_val: pd.Series) -> pd.Series:
    """
    Calculate cluster duration using vectorized operations.
    
    Duration is the number of consecutive bars in the same cluster.
    
    Args:
        cluster_val: Series of cluster values
        
    Returns:
        Series with cluster duration for each timestamp
    """
    if len(cluster_val) == 0:
        return pd.Series(dtype=int, index=cluster_val.index)
    
    # Convert to numpy for faster operations
    cluster_arr = cluster_val.values
    duration_arr = np.zeros(len(cluster_arr), dtype=int)
    
    # Calculate duration: reset to 1 when cluster changes, otherwise increment
    for i in range(len(cluster_arr)):
        if pd.isna(cluster_val.iloc[i]):
            duration_arr[i] = 0
        elif i == 0:
            duration_arr[i] = 1
        elif pd.isna(cluster_val.iloc[i - 1]):
            duration_arr[i] = 1
        elif cluster_arr[i] == cluster_arr[i - 1]:
            duration_arr[i] = duration_arr[i - 1] + 1
        else:
            duration_arr[i] = 1
    
    return pd.Series(duration_arr, index=cluster_val.index)


def vectorized_extreme_duration(
    real_clust: pd.Series,
    extreme_threshold: float,
    max_real_clust: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate extreme duration using vectorized operations.
    
    Args:
        real_clust: Series of real_clust values
        extreme_threshold: Threshold for extreme (0.0-1.0)
        max_real_clust: Maximum real_clust value (1.0 for k=2, 2.0 for k=3)
    
    Returns:
        Tuple of (extreme_duration, in_extreme) Series
    """
    if len(real_clust) == 0:
        return (
            pd.Series(dtype=int, index=real_clust.index),
            pd.Series(dtype=bool, index=real_clust.index),
        )
    
    # Clamp extreme_threshold to valid range to avoid invalid calculations
    # If extreme_threshold > max_real_clust, no values can be extreme
    if extreme_threshold > max_real_clust:
        return (
            pd.Series(0, index=real_clust.index, dtype=int),
            pd.Series(False, index=real_clust.index, dtype=bool),
        )
    
    # Vectorized extreme detection
    is_lower_extreme = (real_clust <= extreme_threshold) & real_clust.notna()
    upper_threshold = max_real_clust - extreme_threshold
    # Only check upper extreme if threshold is valid (upper_threshold >= 0)
    if upper_threshold >= 0:
        is_upper_extreme = (real_clust >= upper_threshold) & real_clust.notna()
    else:
        # If extreme_threshold > max_real_clust/2, upper threshold would be negative
        # In this case, only lower extreme is valid
        is_upper_extreme = pd.Series(False, index=real_clust.index, dtype=bool)
    in_extreme = is_lower_extreme | is_upper_extreme
    
    # Calculate duration: increment when in extreme, reset when not
    extreme_duration = pd.Series(0, index=real_clust.index, dtype=int)
    real_clust_arr = real_clust.values
    in_extreme_arr = in_extreme.values
    
    for i in range(len(real_clust)):
        if in_extreme_arr[i]:
            if i > 0 and in_extreme_arr[i - 1]:
                extreme_duration.iloc[i] = extreme_duration.iloc[i - 1] + 1
            else:
                extreme_duration.iloc[i] = 1
        else:
            extreme_duration.iloc[i] = 0
    
    return extreme_duration, in_extreme


def vectorized_transition_detection(
    prev_cluster: pd.Series,
    curr_cluster: pd.Series,
    bullish_transitions: list[tuple[int, int]],
    bearish_transitions: list[tuple[int, int]],
) -> tuple[pd.Series, pd.Series]:
    """
    Detect cluster transitions using vectorized operations.
    
    Args:
        prev_cluster: Previous cluster values
        curr_cluster: Current cluster values
        bullish_transitions: List of (prev, curr) tuples for bullish transitions
        bearish_transitions: List of (prev, curr) tuples for bearish transitions
        
    Returns:
        Tuple of (bullish_mask, bearish_mask) boolean Series
    """
    # Convert to int, handling NaN
    prev_int = prev_cluster.fillna(-1).astype(int)
    curr_int = curr_cluster.fillna(-1).astype(int)
    
    # Create transition tuples
    transitions = pd.Series(
        list(zip(prev_int, curr_int)),
        index=prev_cluster.index,
        dtype=object,
    )
    
    # Check for bullish transitions
    bullish_mask = pd.Series(False, index=prev_cluster.index)
    for transition in bullish_transitions:
        bullish_mask |= transitions == transition
    
    # Check for bearish transitions
    bearish_mask = pd.Series(False, index=prev_cluster.index)
    for transition in bearish_transitions:
        bearish_mask |= transitions == transition
    
    # Mask out NaN values
    valid_mask = prev_cluster.notna() & curr_cluster.notna()
    bullish_mask &= valid_mask
    bearish_mask &= valid_mask
    
    return bullish_mask, bearish_mask


def vectorized_crossing_detection(
    prev_val: pd.Series,
    curr_val: pd.Series,
    threshold: float,
    direction: str = "up",
) -> pd.Series:
    """
    Detect crossing of threshold using vectorized operations.
    
    Args:
        prev_val: Previous values
        curr_val: Current values
        threshold: Threshold to cross
        direction: "up" for crossing from below, "down" for crossing from above
        
    Returns:
        Boolean Series indicating where crossing occurred
    """
    if direction == "up":
        return (prev_val < threshold) & (curr_val >= threshold) & prev_val.notna() & curr_val.notna()
    else:  # down
        return (prev_val > threshold) & (curr_val <= threshold) & prev_val.notna() & curr_val.notna()

