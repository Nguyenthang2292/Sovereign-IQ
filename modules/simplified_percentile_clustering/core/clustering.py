"""
Main clustering calculation for Simplified Percentile Clustering.

Combines feature calculations, center computation, and cluster assignment
to produce cluster assignments and interpolated cluster values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from modules.simplified_percentile_clustering.core.centers import (
    ClusterCenters,
    compute_centers,
)
from modules.simplified_percentile_clustering.core.features import (
    FeatureCalculator,
    FeatureConfig,
)
from modules.simplified_percentile_clustering.utils.helpers import (
    normalize_cluster_name,
    safe_isna,
    vectorized_min_and_second_min,
    vectorized_min_distance,
)
from modules.simplified_percentile_clustering.utils.validation import (
    validate_clustering_config,
    validate_input_data,
)


@dataclass
class ClusteringConfig:
    """Configuration for clustering calculation."""

    # Clustering parameters
    k: int = 2  # Number of clusters (2 or 3)
    lookback: int = 1000  # Historical bars for percentile/mean calculations
    p_low: float = 5.0  # Lower percentile
    p_high: float = 95.0  # Upper percentile

    # Feature configuration
    feature_config: Optional[FeatureConfig] = None

    # Main plot mode
    main_plot: str = "Clusters"  # "Clusters", "RSI", "CCI", "Fisher", "DMI", "Z-Score", "MAR"

    # Distance calculation improvements
    use_correlation_weights: bool = False  # Weight features by uniqueness (1 - avg_corr)
    time_decay_factor: float = 1.0  # 1.0 = no decay, <1.0 = decay (e.g., 0.99)

    # Interpolation
    interpolation_mode: str = "linear"  # "linear", "sigmoid", "exponential"

    # Cluster stability parameters
    min_flip_duration: int = 3  # Minimum bars in cluster before allowing flip
    flip_confidence_threshold: float = 0.6  # Required confidence to flip clusters

    # Volatility adjustment (for adaptive percentiles)
    volatility_adjustment: bool = False  # Enable volatility-adaptive percentiles

    def __post_init__(self):
        """Validate configuration after initialization."""
        validate_clustering_config(self)


@dataclass
class ClusteringResult:
    """Result of clustering calculation."""

    # Cluster assignment
    cluster_val: pd.Series  # Discrete cluster index (0, 1, or 2)
    curr_cluster: pd.Series  # Cluster name ("k0", "k1", "k2")
    real_clust: pd.Series  # Interpolated cluster value (continuous)

    # Distances
    min_dist: pd.Series  # Distance to closest center
    second_min_dist: pd.Series  # Distance to second closest center
    rel_pos: pd.Series  # Relative position between closest and second closest

    # Plot values
    plot_val: pd.Series  # Value to plot (feature value or real_clust)
    plot_k0_center: pd.Series  # k0 cluster center
    plot_k1_center: pd.Series  # k1 cluster center
    plot_k2_center: pd.Series  # k2 cluster center (if k=3)

    # Feature values (for reference)
    features: dict[str, pd.Series]


class SimplifiedPercentileClustering:
    """Main clustering calculator."""

    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        if self.config.feature_config is None:
            self.config.feature_config = FeatureConfig()

        self.feature_calc = FeatureCalculator(self.config.feature_config)
        self._centers_calculators: dict[str, ClusterCenters] = {}

    def _get_centers_calculator(self, feature_name: str) -> ClusterCenters:
        """Get or create centers calculator for a feature."""
        if feature_name not in self._centers_calculators:
            self._centers_calculators[feature_name] = ClusterCenters(
                lookback=self.config.lookback,
                p_low=self.config.p_low,
                p_high=self.config.p_high,
                k=self.config.k,
            )
        return self._centers_calculators[feature_name]

    def _compute_all_centers(self, features: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
        """
        Compute cluster centers for all features using vectorized operations.

        Uses the vectorized compute_centers() function instead of iterative updates.
        """
        centers_dict = {}

        for feature_name, values in features.items():
            if feature_name.endswith("_val") or feature_name in ["zsc_val"]:
                # Use vectorized compute_centers instead of loop
                centers_df = compute_centers(
                    values,
                    lookback=self.config.lookback,
                    p_low=self.config.p_low,
                    p_high=self.config.p_high,
                    k=self.config.k,
                    volatility_adjustment=getattr(self.config, "volatility_adjustment", False),
                )
                centers_dict[feature_name] = centers_df

        return centers_dict

    def _compute_distance_single(self, feature_val: pd.Series, centers: pd.DataFrame) -> pd.Series:
        """
        Compute distance for single-feature mode using vectorized operations.

        This method uses vectorized operations for better performance.
        """
        return vectorized_min_distance(feature_val, centers)

    def _compute_distance_combined(
        self,
        features: dict[str, pd.Series],
        centers_dict: dict[str, pd.DataFrame],
        center_idx: int,
    ) -> pd.Series:
        """
        Compute combined distance across all enabled features using vectorized operations.

        Supports correlation-based weighting and time decay.
        """
        config = self.config.feature_config
        center_col = f"k{center_idx}"

        # 1. Identify active features based on config and availability
        potential_features = [
            ("use_rsi", "rsi_val"),
            ("use_cci", "cci_val"),
            ("use_fisher", "fisher_val"),
            ("use_dmi", "dmi_val"),
            ("use_zscore", "zsc_val"),
            ("use_mar", "mar_val"),
        ]

        active_keys = []
        for conf_attr, feat_key in potential_features:
            if getattr(config, conf_attr) and feat_key in features:
                # Also check if centers exist for this feature
                if feat_key in centers_dict and center_col in centers_dict[feat_key]:
                    active_keys.append(feat_key)

        if not active_keys:
            # Return NaN series if no features enabled/available
            index = next(iter(features.values())).index if features else pd.Index([])
            return pd.Series(np.nan, index=index)

        # 2. Compute Raw Distances (DataFrame)
        # Use index from the first available feature
        index = features[active_keys[0]].index
        dist_df = pd.DataFrame(index=index)

        for key in active_keys:
            dist_df[key] = (features[key] - centers_dict[key][center_col]).abs()

        # 3. Compute Feature Weights
        # Default: Equal weights (1.0)
        weights = pd.Series(1.0, index=active_keys)

        if self.config.use_correlation_weights and len(active_keys) > 1:
            # Get feature values for correlation matrix
            feat_vals = pd.DataFrame({k: features[k] for k in active_keys}, index=index)
            # Calculate correlation matrix
            corr_matrix = feat_vals.corr().abs()

            uniqueness = {}
            for col in active_keys:
                # Average correlation with OTHER features
                other_cols = [c for c in active_keys if c != col]
                if other_cols:
                    avg_corr = corr_matrix.loc[col, other_cols].mean()
                else:
                    avg_corr = 0.0
                uniqueness[col] = 1.0 - avg_corr

            # Create weights series
            u_series = pd.Series(uniqueness)
            if u_series.sum() > 0:
                weights = u_series
                # Note: We don't necessarily need to normalize to sum=1 here
                # because we divide by sum of weights later (Weighted Mean).

        # 4. Calculate Weighted Mean across Features
        # Formula: Sum(Dist_i * W_i) / Sum(W_i for valid Dist_i)

        # Multiply distances by weights (broadcasting over columns)
        weighted_dists = dist_df * weights

        # Sum weighted distances per row
        numer = weighted_dists.sum(axis=1, skipna=True)

        # Calculate sum of weights for valid (non-NaN) distances per row
        valid_mask = dist_df.notna()
        # Broadcast weights to mask
        valid_weights = valid_mask * weights
        denom = valid_weights.sum(axis=1)

        # Calculate combined weighted distance (handling division by zero)
        combined_dist = numer / denom.replace(0, np.nan)

        # 5. Apply Time Decay to Final Result
        if self.config.time_decay_factor < 1.0:
            n = len(index)
            decay = self.config.time_decay_factor
            # Generate exponents: n-1 down to 0
            # Recent (last index) gets decay^0 = 1
            # Oldest (first index) gets decay^(n-1)
            exponents = np.arange(n)[::-1]
            time_decay_mult = pd.Series(np.power(decay, exponents), index=index)

            combined_dist = combined_dist * time_decay_mult

        return combined_dist

    def _compute_real_clust_nonlinear(
        self,
        cluster_val: pd.Series,
        second_val: pd.Series,
        min_dist: pd.Series,
        second_min_dist: pd.Series,
        interpolation_mode: str = "linear",
    ) -> tuple[pd.Series, pd.Series]:
        """
        Non-linear interpolation for smoother cluster transitions.

        Modes:
        - linear: Current behavior (linear interpolation)
        - sigmoid: S-curve (smooth transitions near centers, sharp in middle)
        - exponential: Exponential decay (sticky to current cluster)
        """
        # Calculate base relative position
        rel_pos = pd.Series(0.0, index=cluster_val.index)

        # Handle edge case where both distances are zero
        both_zero = (min_dist == 0) & (second_min_dist == 0) & (~safe_isna(min_dist)) & (~safe_isna(second_min_dist))
        rel_pos[both_zero] = 0.5

        # Handle normal case
        valid_rel = (second_min_dist > 0) & (second_min_dist != np.inf) & (~safe_isna(second_min_dist)) & (~both_zero)

        # Base calculation (Linear)
        # Avoid division by zero by filtering with valid_rel
        if valid_rel.any():
            base_rel_pos = min_dist[valid_rel] / (min_dist[valid_rel] + second_min_dist[valid_rel])

            # Apply non-linear transformation
            if interpolation_mode == "sigmoid":
                # Sigmoid: S-curve centered at 0.5
                # Maps [0, 1] -> [0, 1] with smooth transitions
                # Using coefficient 10 for steepness
                transformed = 1.0 / (1.0 + np.exp(-10.0 * (base_rel_pos - 0.5)))
                rel_pos[valid_rel] = transformed

            elif interpolation_mode == "exponential":
                # Exponential decay: sticky to current cluster
                # Maps [0, 1] -> [0, ~0.95] with emphasis on staying in current cluster
                transformed = 1.0 - np.exp(-3.0 * base_rel_pos)
                rel_pos[valid_rel] = transformed

            else:  # linear
                rel_pos[valid_rel] = base_rel_pos

        # Interpolated cluster
        real_clust = cluster_val + (second_val - cluster_val) * rel_pos
        return real_clust, rel_pos

    def _calculate_confidence(self, rel_pos: pd.Series) -> pd.Series:
        """
        Calculate confidence score based on relative position.

        rel_pos ranges from 0.0 (center) to 0.5 (midpoint).
        Confidence = 1.0 - (2.0 * rel_pos)
        Maps [0.0, 0.5] -> [1.0, 0.0]
        """
        return 1.0 - (2.0 * rel_pos)

    def _apply_cluster_stability(
        self,
        curr_cluster: pd.Series,
        confidence: pd.Series,
    ) -> pd.Series:
        """
        Apply stability rules to prevent rapid cluster flipping.

        Rules:
        1. Must be in current cluster for min_flip_duration bars
        2. New cluster must have confidence > threshold
        """
        min_duration = self.config.min_flip_duration
        conf_threshold = self.config.flip_confidence_threshold

        # If stability not required, return raw cluster
        if min_duration <= 1 and conf_threshold <= 0:
            return curr_cluster

        n = len(curr_cluster)
        stable_cluster = curr_cluster.values.copy()  # Use numpy array for speed
        raw_cluster_vals = curr_cluster.values
        confidence_vals = confidence.values.flatten() if isinstance(confidence, pd.DataFrame) else confidence.values

        # Track state
        current_state = raw_cluster_vals[0]
        duration = 1

        for i in range(1, n):
            new_val = raw_cluster_vals[i]

            if new_val == current_state:
                # Same cluster, increment duration
                duration += 1
                stable_cluster[i] = current_state
            else:
                # Attempt to flip
                # Check stability conditions based on stable state
                # Note: 'duration' tracks how long we've been in 'current_state'
                is_stable = duration >= min_duration and confidence_vals[i] >= conf_threshold

                if is_stable:
                    # Allow flip
                    current_state = new_val
                    duration = 1
                    stable_cluster[i] = current_state
                else:
                    # Reject flip, stay in current state
                    stable_cluster[i] = current_state
                    duration += 1  # We stayed in the same state effectively

        return pd.Series(stable_cluster, index=curr_cluster.index)

    def compute(self, high: pd.Series, low: pd.Series, close: pd.Series) -> ClusteringResult:
        """
        Compute clustering for OHLCV data.

        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.

        Returns:
            ClusteringResult with all computed values.

        Raises:
            ValueError: If input data is invalid
        """
        # Validate input data
        validate_input_data(high=high, low=low, close=close, require_all=True)

        # Step 1: Compute all features
        features = self.feature_calc.compute_all(high, low, close, self.config.lookback)

        # Step 2: Compute centers for all features
        centers_dict = self._compute_all_centers(features)

        # Step 3: Determine which feature/centers to use based on main_plot
        main_plot = self.config.main_plot
        index = close.index
        len(close)

        # Step 4: Compute distances to all centers using vectorized operations
        # Build a matrix of distances: rows = timestamps, columns = centers (k0, k1, k2)
        distances_df = pd.DataFrame(index=index)

        # Map main_plot to feature key
        feature_map = {
            "RSI": "rsi_val",
            "CCI": "cci_val",
            "Fisher": "fisher_val",
            "DMI": "dmi_val",
            "Z-Score": "zsc_val",
            "MAR": "mar_val",
        }

        # Compute distances for each center using vectorized operations
        for center_idx in range(self.config.k):
            center_col = f"k{center_idx}"

            if main_plot in feature_map:
                # Single feature mode - use broadcasting
                feature_key = feature_map[main_plot]
                if feature_key in features and feature_key in centers_dict:
                    feature_vals = features[feature_key]
                    centers = centers_dict[feature_key]
                    if center_col in centers.columns:
                        # Vectorized distance calculation: abs(feature_series - center_series)
                        dist = (feature_vals - centers[center_col]).abs()
                        distances_df[center_col] = dist
                    else:
                        distances_df[center_col] = pd.Series(np.nan, index=index)
                else:
                    distances_df[center_col] = pd.Series(np.nan, index=index)
            else:
                # Combined mode - use vectorized combined distance
                dist_series = self._compute_distance_combined(features, centers_dict, center_idx)
                distances_df[center_col] = dist_series

        # Replace inf with NaN for easier handling
        distances_df = distances_df.replace([np.inf, -np.inf], np.nan)

        # Step 5: Find min and second min distances using vectorized operations
        # Convert to numpy array for efficient operations
        dist_array = distances_df.values

        # Use helper function for vectorized min and second min calculation
        min_dist_arr, second_min_dist_arr, cluster_val_arr, second_cluster_val_arr = vectorized_min_and_second_min(
            dist_array
        )

        # Convert to Series
        min_dist = pd.Series(min_dist_arr, index=index)
        second_min_dist = pd.Series(second_min_dist_arr, index=index)
        cluster_val = pd.Series(cluster_val_arr, index=index)

        # Convert cluster values to cluster names using helper function
        curr_cluster = pd.Series([normalize_cluster_name(cv) for cv in cluster_val_arr], index=index, dtype=object)
        second_cluster = pd.Series(
            [normalize_cluster_name(cv) for cv in second_cluster_val_arr], index=index, dtype=object
        )

        # Second cluster value - convert cluster names to numeric values
        second_val = pd.Series(cluster_val.values, index=index)
        second_val_mask = second_cluster.notna()
        second_val[second_cluster == "k0"] = 0.0
        second_val[second_cluster == "k1"] = 1.0
        second_val[second_cluster == "k2"] = 2.0
        # If second_cluster is None/NaN, use cluster_val
        second_val[~second_val_mask] = cluster_val[~second_val_mask]

        # --- Cluster Stability Enhancement ---
        # 1. Calculate base rel_pos for confidence (using linear just to get raw position 0-0.5)
        # We need this before stability to check if we SHOULD flip

        # Re-calculate linear rel_pos for confidence
        raw_rel_pos = pd.Series(0.0, index=index)
        both_zero = (min_dist == 0) & (second_min_dist == 0) & (~safe_isna(min_dist)) & (~safe_isna(second_min_dist))
        raw_rel_pos[both_zero] = 0.5

        valid_rel = (second_min_dist > 0) & (second_min_dist != np.inf) & (~safe_isna(second_min_dist)) & (~both_zero)
        if valid_rel.any():
            raw_rel_pos[valid_rel] = min_dist[valid_rel] / (min_dist[valid_rel] + second_min_dist[valid_rel])

        # 2. Calculate confidence
        confidence = self._calculate_confidence(raw_rel_pos)

        # 3. Apply stability (flip dampening)
        # Refines cluster_val based on history and confidence
        stable_cluster_val = self._apply_cluster_stability(cluster_val, confidence)

        # Update cluster_val to stable version
        cluster_val = stable_cluster_val

        # 4. Update second_val based on STABLE cluster_val (if needed)
        # logic: if cluster_val changed (flipped back), second_val must be the "other" one.
        # But wait, second_val was derived from "second_cluster" which was based on raw distances.
        # If we FORCE cluster_val to stay at 0, but raw distances say closest=1, second=0.
        # Then stable_cluster=0. Raw closest=1. Raw second=0.
        # So "second_val" (alternative) should probably be the RAW closest (1).
        # Efficient approximation: If stable_cluster != raw_cluster, then raw_cluster IS the "second best" (actually the best, but we ignored it).
        # If stable_cluster == raw_cluster, then second_val is just second_cluster.

        # Let's simple re-evaluate second_val:
        # If stable != raw (we dampened a flip):
        #   Current = stable (old value).
        #   Target (that we ignored) = raw (new value).
        #   So "second" should be raw.
        # If stable == raw (no dampening or valid flip):
        #   Current = raw.
        #   Second = raw_second.

        # Recalculate second_val
        # raw_cluster_val_arr was the original. Let's access it via cluster_val_arr from earlier scope if possible,
        # but we overwrote cluster_val. Let's use the property that if stable != raw (implicit in distances), take raw.
        # Actually, we can reconstruct raw from min/second/cluster_val logic, BUT
        # simpler approach:
        # The 'second_val' passed to interpolation defines the "direction" of rel_pos.
        # rel_pos = min / (min + second). This assumes min corresponds to cluster_val.
        # IF we forced cluster_val to be the "farther" one (dampening), then "min_dist" is actually the distance to the dampened center?
        # NO. cluster_val should point to the center we are assigned to.
        # If we assign to stable_cluster (which is NOT min_dist anymore), we technically have a "negative" rel_pos or > 0.5 rel_pos relative to min_dist?
        # Actually, interpolation assumes we are moving FROM cluster_val TO second_val.
        # If we are stuck in 'old' cluster, but price is moving to 'new' cluster:
        # cluster_val = old. second_val = new.
        # rel_pos should increase from 0 to 1.
        # min_dist (to new) < dist (to old).
        # So rel_pos calculation (min / min+sec) gives < 0.5.
        # This implies we are close to NEW.
        # But we want `real_clust` to reflect that we are "almost" at new, but snapped to OLD?
        # NO. `real_clust` is a continuous value.
        # If we dampen the DISCRETE flip, `real_clust` should probably still move continuously?
        # OR should `real_clust` also be dampened?
        # Usually, `real_clust` tracks the continuous drift. The discrete `cluster_val` is the trading signal.
        # IF we change `cluster_val` but leave `rel_pos` as is (based on min_dist), `real_clust` calculation:
        # real = cluster + (second - cluster) * rel_pos.
        # If cluster=0, second=1. rel_pos=0.1 (close to 1?? No, min_dist is to 1. So rel_pos is dist to 1. Small.)
        # wait. rel_pos = min / (min + sec). min is dist to CLOSEST.
        # If closest is 1. dist(1)=10. dist(0)=90. rel_pos = 10/100 = 0.1.
        # If we say cluster=0. Then we are at 0. But we are actually close to 1.
        # Formula: 0 + (1 - 0) * 0.1 = 0.1. -> Close to 0. THIS IS WRONG. We are close to 1.
        # If we forced cluster=0, we are "far" from 0.
        # Ideally, we want real_clust to show 0.9 (close to 1).
        # So if we force cluster=0, we need rel_pos to be 0.9.
        # But rel_pos is derived from min_dist (to 1).
        # The variables `min_dist`, `second_min_dist` match the RAW closest.

        # FIX:
        # If stable_cluster != raw_cluster (closest):
        # Then "cluster_val" is NOT the closest.
        # "second_val" should be the closest.
        # "rel_pos" (dist to closest / total) is small (e.g. 0.1).
        # We start at stable (0), go to closest (1).
        # We are at 0.9 distance from 0?
        # Distance to 0 is `second_min_dist` (large). Distance to 1 is `min_dist` (small).
        # We want real_clust = 0.9.
        # Formula: stable + (closest - stable) * (dist_to_stable / (dist_to_stable + dist_to_closest))
        # = 0 + (1 - 0) * (90 / 100) = 0.9.
        # So effective_rel_pos = second_min_dist / (min + second).
        # = 1 - (min / (min + second)) = 1 - raw_rel_pos.

        # Correct Logic:
        # 1. raw_cluster corresponds to min_dist.
        # 2. stable_cluster is what we chose.
        # 3. If stable == raw:
        #    primary = min_dist. secondary = second_min_dist. target = raw_second.
        #    rel_pos = min / (min + sec).
        # 4. If stable != raw:
        #    primary = second_min_dist (dist to stable). secondary = min_dist (dist to raw/target).
        #    target = raw_cluster.
        #    rel_pos = primary / (prim + sec) = second / (min + sec).

        # We need to construct aligned vectors for the interpolation call.

        # Get raw closest cluster (normalized names/values)
        raw_cluster_numeric = pd.Series(cluster_val_arr, index=index)

        # Identify where we kept old value despite raw wanting to flip
        # (stable != raw)
        is_dampened = stable_cluster_val != raw_cluster_numeric

        # Prepare inputs for interpolation
        # Default: current = stable (which is raw), second = raw_second, d_curr = min, d_target = second_min

        # Vectors to pass to _compute
        i_cluster_val = stable_cluster_val.copy()

        # If dampened:
        # cluster_val is stable (old).
        # second_val is raw (new).
        # d_curr (to stable) is second_min_dist.
        # d_target (to raw) is min_dist.

        # If NOT dampened:
        # cluster_val is stable (raw).
        # second_val is raw_second.
        # d_curr (to stable) is min_dist.
        # d_target (to raw_second) is second_min_dist.

        i_second_val = second_val.copy()
        i_min_dist = min_dist.copy()  # Dist to i_cluster_val
        i_second_min_dist = second_min_dist.copy()  # Dist to i_second_val

        if is_dampened.any():
            # Adjust vectors where dampened
            # second_val becomes raw_cluster
            # (Note: raw_cluster is numeric, second_val is numeric)
            i_second_val[is_dampened] = raw_cluster_numeric[is_dampened]

            # Swapping distances because "min_dist" was to raw, but now i_cluster_val is "far" (second_min)
            # dist to i_cluster_val (stable) is actually the larger one (second_min_dist)
            i_min_dist[is_dampened] = second_min_dist[is_dampened]
            # dist to i_second_val (raw/target) is the smaller one (min_dist)
            i_second_min_dist[is_dampened] = min_dist[is_dampened]

        # Call interpolation with ADJUSTED vectors
        real_clust, rel_pos = self._compute_real_clust_nonlinear(
            i_cluster_val,
            i_second_val,
            i_min_dist,
            i_second_min_dist,
            interpolation_mode=self.config.interpolation_mode,
        )

        # Since we modified cluster_val to stable_cluster_val, we should also update
        # curr_cluster and second_cluster for the result object if we want consistency?
        # The result object has 'curr_cluster' (string).
        # We should update it to match stable_cluster_val.

        # Map numeric stable_cluster_val back to "k0", "k1", etc.
        # (Assuming k=2 or 3 is simple mapping)
        def map_val_to_name(v):
            if v == 0:
                return "k0"
            if v == 1:
                return "k1"
            if v == 2:
                return "k2"
            return str(v)

        # Vectorized map?
        # stable_cluster_val is float/int series
        curr_cluster = stable_cluster_val.map(map_val_to_name)

        # Compute plot values
        if main_plot == "Clusters":
            plot_val = real_clust
        elif main_plot == "RSI" and "rsi_val" in features:
            plot_val = features["rsi_val"]
        elif main_plot == "CCI" and "cci_val" in features:
            plot_val = features["cci_val"]
        elif main_plot == "Fisher" and "fisher_val" in features:
            plot_val = features["fisher_val"]
        elif main_plot == "DMI" and "dmi_val" in features:
            plot_val = features["dmi_val"]
        elif main_plot == "Z-Score" and "zsc_val" in features:
            plot_val = features["zsc_val"]
        elif main_plot == "MAR" and "mar_val" in features:
            plot_val = features["mar_val"]
        else:
            plot_val = real_clust

        # Compute plot centers
        if main_plot == "Clusters":
            plot_k0_center = pd.Series(0.0, index=index)
            plot_k1_center = pd.Series(1.0, index=index)
            plot_k2_center = pd.Series(2.0, index=index) if self.config.k == 3 else pd.Series(0.0, index=index)
        else:
            # Use centers from the selected feature
            feature_key = {
                "RSI": "rsi_val",
                "CCI": "cci_val",
                "Fisher": "fisher_val",
                "DMI": "dmi_val",
                "Z-Score": "zsc_val",
                "MAR": "mar_val",
            }.get(main_plot)

            if feature_key and feature_key in centers_dict:
                centers = centers_dict[feature_key]
                plot_k0_center = centers["k0"]
                plot_k1_center = centers["k1"]
                plot_k2_center = centers["k2"] if self.config.k == 3 else pd.Series(0.0, index=index)
            else:
                plot_k0_center = pd.Series(0.0, index=index)
                plot_k1_center = pd.Series(0.0, index=index)
                plot_k2_center = pd.Series(0.0, index=index)

        return ClusteringResult(
            cluster_val=cluster_val,
            curr_cluster=curr_cluster,
            real_clust=real_clust,
            min_dist=min_dist,
            second_min_dist=second_min_dist,
            rel_pos=rel_pos,
            plot_val=plot_val,
            plot_k0_center=plot_k0_center,
            plot_k1_center=plot_k1_center,
            plot_k2_center=plot_k2_center,
            features=features,
        )


def compute_clustering(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    config: Optional[ClusteringConfig] = None,
) -> ClusteringResult:
    """
    Convenience function to compute clustering.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        config: ClusteringConfig instance (optional).

    Returns:
        ClusteringResult with all computed values.

    Raises:
        ValueError: If input data or config is invalid
    """
    clustering = SimplifiedPercentileClustering(config)
    return clustering.compute(high, low, close)


def compute_multi_timeframe_clustering(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeframes: list[str] = ["1h", "4h", "1d"],
    require_alignment: bool = True,
    config: Optional[ClusteringConfig] = None,
) -> dict:
    """
    Multi-timeframe clustering for stronger signals.

    Args:
        high: Source high price series (e.g. 1m or 5m data)
        low: Source low price series
        close: Source close price series
        timeframes: List of timeframes to analyze (e.g., ["1h", "4h"])
        require_alignment: If True, all timeframes must agree on cluster for 'aligned_cluster'

    Returns:
        Dict with:
        - cluster_val: Primary timeframe cluster (timeframes[0])
        - mtf_agreement: Fraction of timeframes agreeing (0.0-1.0)
        - aligned_cluster: Cluster if all timeframes agree, else None/NaN
        - timeframe_results: Dict of full clustering results for each timeframe
    """
    # Validate input index
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("Input data must have a DatetimeIndex for resampling.")

    results = {}

    # Analyze each timeframe
    for tf in timeframes:
        # Resample to timeframe
        # Note: We assume the input is lower or equal resolution than the target timeframes.
        # e.g. input is 15m, target is 1h, 4h.
        try:
            ohlc_dict = {
                "high": high.resample(tf).max(),
                "low": low.resample(tf).min(),
                "close": close.resample(tf).last(),
            }
        except ValueError as e:
            # Handle invalid timeframe strings
            raise ValueError(f"Invalid timeframe string '{tf}': {e}")

        # Drop NaNs created by resampling (if any)
        ohlc_dict["high"] = ohlc_dict["high"].dropna()
        ohlc_dict["low"] = ohlc_dict["low"].dropna()
        ohlc_dict["close"] = ohlc_dict["close"].dropna()

        if len(ohlc_dict["close"]) == 0:
            continue

        # Compute clustering with provided config
        clustering = SimplifiedPercentileClustering(config)
        result = clustering.compute(ohlc_dict["high"], ohlc_dict["low"], ohlc_dict["close"])

        # Align back to original timeframe (ffill)
        # Reindex to match the INPUT close index
        aligned_series = result.cluster_val.reindex(close.index, method="ffill")
        results[tf] = aligned_series

    if not results:
        return {
            "cluster_val": pd.Series(np.nan, index=close.index),
            "mtf_agreement": 0.0,
            "aligned_cluster": pd.Series(np.nan, index=close.index),
            "timeframe_results": {},
        }

    # Calculate agreement based on Primary Timeframe (first in list)
    primary_tf = timeframes[0]
    if primary_tf not in results:
        # Fallback if primary failed
        primary_tf = next(iter(results.keys()))

    primary_cluster = results[primary_tf]

    # Calculate agreement score (Global Average)
    # Average agreement of other TFs with Primary
    agreement_scores = []
    other_tfs = [tf for tf in timeframes if tf != primary_tf and tf in results]

    if other_tfs:
        for tf in other_tfs:
            agreement_mask = results[tf] == primary_cluster
            score = agreement_mask.sum() / len(primary_cluster)
            agreement_scores.append(score)
        mtf_agreement = np.mean(agreement_scores)
    else:
        mtf_agreement = 1.0

    # Determine Aligned Cluster (Bar-by-Bar)
    # Start with None/NaN
    aligned_cluster = pd.Series(np.nan, index=close.index)

    # Intersection logic: Where ALL timeframes agree
    # Start mask as True
    agreement_mask = pd.Series(True, index=close.index)

    # Check alignment across ALL computed results
    for tf in results:
        agreement_mask &= results[tf] == primary_cluster

    # Where agreed, use primary value
    aligned_cluster[agreement_mask] = primary_cluster[agreement_mask]

    return {
        "cluster_val": primary_cluster,
        "mtf_agreement": mtf_agreement,
        "aligned_cluster": aligned_cluster,
        "timeframe_results": results,
    }


__all__ = [
    "ClusteringConfig",
    "ClusteringResult",
    "SimplifiedPercentileClustering",
    "compute_clustering",
    "compute_multi_timeframe_clustering",
]
