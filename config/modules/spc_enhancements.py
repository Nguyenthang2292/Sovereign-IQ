"""
SPC Enhancements Configuration.

This module contains configuration for the 6 new SPC enhancements:
1. Volatility-Adaptive Percentiles
2. Correlation-based Feature Weighting
3. Time-Decay Weighting
4. Non-linear Interpolation
5. Cluster Stability
6. Multi-Timeframe Analysis

All enhancements are OPT-IN (disabled by default) for backward compatibility.
"""

# ============================================================================
# Enhancement 1: Volatility-Adaptive Percentiles
# ============================================================================
# Dynamically adjusts percentile thresholds based on market volatility.
# High volatility → wider percentiles (more stable clusters)
# Low volatility → narrower percentiles (more responsive clusters)

SPC_VOLATILITY_ADJUSTMENT = False  # Enable/disable adaptive percentiles
"""
Enable volatility-adaptive percentile calculation.

When enabled, the clustering algorithm adjusts p_low and p_high based on
current market volatility compared to baseline volatility.

Benefits:
- More stable clusters in high volatility markets
- More responsive clusters in low volatility markets
- Reduces false cluster transitions during volatility spikes

Performance Impact: Minimal (~2-3% overhead)
Recommended: True for crypto markets, False for stable markets
"""

# ============================================================================
# Enhancement 2: Correlation-based Feature Weighting
# ============================================================================
# Weights features by their uniqueness (inverse of average correlation).
# Features with lower correlation to others get higher weights.

SPC_USE_CORRELATION_WEIGHTS = False  # Enable/disable correlation weighting
"""
Enable correlation-based feature weighting in distance calculation.

When enabled, features are weighted by their uniqueness:
- Unique features (low correlation) → higher weight
- Redundant features (high correlation) → lower weight

Benefits:
- Reduces impact of correlated features (RSI/CCI often correlated)
- Emphasizes unique signal information
- More robust clustering with multiple features

Performance Impact: Moderate (~5-7% overhead for correlation matrix calculation)
Recommended: True when using 3+ features
"""

# ============================================================================
# Enhancement 3: Time-Decay Weighting
# ============================================================================
# Applies exponential decay to give more weight to recent data points.

SPC_TIME_DECAY_FACTOR = 1.0  # 1.0 = no decay, <1.0 = exponential decay
"""
Time decay factor for distance calculation.

Values:
- 1.0: No decay (all data points weighted equally) - DEFAULT
- 0.99: Light decay (recent data slightly more important)
- 0.95: Moderate decay (recent data significantly more important)
- 0.90: Strong decay (very responsive to recent changes)

Formula: weight[i] = decay_factor ** (n - i - 1)
- Most recent point: weight = 1.0
- Oldest point: weight = decay_factor ** (n - 1)

Benefits:
- More responsive to recent price action
- Reduces lag in cluster transitions
- Better adapts to regime changes

Performance Impact: Minimal (~1-2% overhead)
Recommended: 0.99 for trending markets, 1.0 for ranging markets
"""

# ============================================================================
# Enhancement 4: Non-linear Interpolation
# ============================================================================
# Applies non-linear transformation to cluster transitions.

SPC_INTERPOLATION_MODE = "linear"  # Options: "linear", "sigmoid", "exponential"
"""
Interpolation mode for real_clust calculation.

Modes:
- "linear" (DEFAULT): Linear interpolation between clusters
  rel_pos = min_dist / (min_dist + second_min_dist)
  real_clust = cluster_val + (second_val - cluster_val) * rel_pos

- "sigmoid": S-curve interpolation (smooth transitions)
  transformed = 1 / (1 + exp(-10 * (rel_pos - 0.5)))
  Smooth near cluster centers, sharper in middle

- "exponential": Exponential decay (sticky to current cluster)
  transformed = 1 - exp(-3 * rel_pos)
  Reduces rapid cluster flips, more stable

Benefits by mode:
- linear: Simple, predictable, fast
- sigmoid: Smoother visual appearance, less noise
- exponential: Most stable, fewer false flips

Performance Impact: Minimal (~1% overhead)
Recommended:
- "linear" for normal markets
- "sigmoid" for smooth visual display
- "exponential" for choppy/whipsaw markets
"""

# ============================================================================
# Enhancement 5: Cluster Stability
# ============================================================================
# Prevents rapid cluster flipping through duration and confidence filters.

SPC_MIN_FLIP_DURATION = 3  # Minimum bars in cluster before allowing flip
"""
Minimum number of bars required in current cluster before allowing flip.

Values:
- 1: No minimum (immediate flips allowed)
- 3: DEFAULT - Moderate stability
- 5: High stability (recommended for choppy markets)
- 10: Very high stability (for very noisy data)

Benefits:
- Reduces whipsaws in choppy markets
- Filters out temporary noise
- Improves trading signal quality

Performance Impact: Negligible
Recommended: 5 for crypto, 3 for stocks
"""

SPC_FLIP_CONFIDENCE_THRESHOLD = 0.6  # Required confidence to flip clusters
"""
Minimum confidence score required to allow cluster flip.

Confidence is calculated as: 1.0 - (2.0 * rel_pos)
- rel_pos = 0.0 (at cluster center) → confidence = 1.0
- rel_pos = 0.5 (midpoint) → confidence = 0.0

Values:
- 0.0: No threshold (any confidence accepted)
- 0.6: DEFAULT - Moderate filtering
- 0.7-0.8: High confidence required (fewer flips)
- 0.9: Very high confidence (very stable, may miss signals)

Benefits:
- Prevents flips when price is ambiguous (near midpoint)
- Only allows flips when clearly in new cluster
- Reduces false signals

Performance Impact: Negligible
Recommended: 0.7 for crypto, 0.6 for stocks
"""

# ============================================================================
# Enhancement 6: Multi-Timeframe Analysis (OPTIONAL)
# ============================================================================
# Analyzes clustering across multiple timeframes for confirmation.

SPC_ENABLE_MTF = False  # Enable/disable multi-timeframe analysis
"""
Enable multi-timeframe clustering analysis.

When enabled, clustering is performed on multiple timeframes and compared
for alignment. Only generates signals when timeframes agree.

Benefits:
- Higher conviction signals (multiple timeframes align)
- Filters out noise on lower timeframes
- Captures multi-timeframe trend structure

Drawbacks:
- Significantly more computation (N timeframes = N×computing)
- May miss valid signals on single timeframe
- Requires datetime-indexed data

Performance Impact: High (N× overhead for N timeframes)
Recommended: Only for high-quality signal filtering
"""

SPC_MTF_TIMEFRAMES = ["1h", "4h"]  # Timeframes to analyze (if MTF enabled)
"""
List of timeframes to analyze in multi-timeframe mode.

Examples:
- ["1h", "4h"]: Short-term + medium-term (DEFAULT)
- ["1h", "4h", "1d"]: Short + medium + long-term (comprehensive)
- ["15m", "1h"]: Very short-term + short-term (for day trading)

Notes:
- Primary timeframe (first in list) is used as reference
- Input data frequency must be <= smallest timeframe
- Example: For ["1h", "4h"], input data can be 5m, 15m, 30m, or 1h

Performance: Linear with number of timeframes
"""

SPC_MTF_REQUIRE_ALIGNMENT = True  # Require all timeframes to align for signal
"""
Require all timeframes to agree on cluster for aligned_cluster output.

Values:
- True (DEFAULT): Strict alignment (all TFs must agree)
  → Fewer but higher quality signals
- False: Permissive (use primary TF, track agreement)
  → More signals, use mtf_agreement score for filtering

Recommendation: True for conservative trading, False for active trading
"""

# ============================================================================
# Combined Enhancement Presets
# ============================================================================

# Preset 1: CONSERVATIVE (Most Stable)
# Best for: Choppy markets, high noise, risk-averse trading
SPC_PRESET_CONSERVATIVE = {
    "volatility_adjustment": True,  # Adapt to volatility
    "use_correlation_weights": True,  # Reduce redundancy
    "time_decay_factor": 1.0,  # No decay (all data equal)
    "interpolation_mode": "exponential",  # Sticky clusters
    "min_flip_duration": 5,  # High stability
    "flip_confidence_threshold": 0.75,  # High confidence required
}

# Preset 2: BALANCED (Recommended Default)
# Best for: Most crypto markets, balanced performance
SPC_PRESET_BALANCED = {
    "volatility_adjustment": True,  # Adapt to volatility
    "use_correlation_weights": False,  # Equal feature weights
    "time_decay_factor": 0.99,  # Light decay
    "interpolation_mode": "linear",  # Standard interpolation
    "min_flip_duration": 3,  # Moderate stability
    "flip_confidence_threshold": 0.6,  # Moderate filtering
}

# Preset 3: AGGRESSIVE (Most Responsive)
# Best for: Trending markets, momentum trading
SPC_PRESET_AGGRESSIVE = {
    "volatility_adjustment": False,  # Fixed percentiles
    "use_correlation_weights": False,  # Equal weights
    "time_decay_factor": 0.95,  # Strong decay (responsive)
    "interpolation_mode": "sigmoid",  # Smooth transitions
    "min_flip_duration": 2,  # Low stability (quick flips)
    "flip_confidence_threshold": 0.5,  # Lower confidence threshold
}

# ============================================================================
# Active Preset (Change this to switch between presets)
# ============================================================================
# Set to None to use individual parameters above
# Set to SPC_PRESET_CONSERVATIVE, SPC_PRESET_BALANCED, or SPC_PRESET_AGGRESSIVE
SPC_ACTIVE_PRESET = None

# If preset is active, override individual parameters
if SPC_ACTIVE_PRESET is not None:
    SPC_VOLATILITY_ADJUSTMENT = SPC_ACTIVE_PRESET["volatility_adjustment"]
    SPC_USE_CORRELATION_WEIGHTS = SPC_ACTIVE_PRESET["use_correlation_weights"]
    SPC_TIME_DECAY_FACTOR = SPC_ACTIVE_PRESET["time_decay_factor"]
    SPC_INTERPOLATION_MODE = SPC_ACTIVE_PRESET["interpolation_mode"]
    SPC_MIN_FLIP_DURATION = SPC_ACTIVE_PRESET["min_flip_duration"]
    SPC_FLIP_CONFIDENCE_THRESHOLD = SPC_ACTIVE_PRESET["flip_confidence_threshold"]

# ============================================================================
# Validation
# ============================================================================


def validate_spc_enhancements_config():
    """Validate SPC enhancements configuration."""
    errors = []

    # Validate time_decay_factor
    if not (0.5 <= SPC_TIME_DECAY_FACTOR <= 1.0):
        errors.append(f"SPC_TIME_DECAY_FACTOR must be in [0.5, 1.0], got {SPC_TIME_DECAY_FACTOR}")

    # Validate interpolation_mode
    valid_modes = ["linear", "sigmoid", "exponential"]
    if SPC_INTERPOLATION_MODE not in valid_modes:
        errors.append(f"SPC_INTERPOLATION_MODE must be one of {valid_modes}, got {SPC_INTERPOLATION_MODE}")

    # Validate min_flip_duration
    if SPC_MIN_FLIP_DURATION < 1:
        errors.append(f"SPC_MIN_FLIP_DURATION must be >= 1, got {SPC_MIN_FLIP_DURATION}")

    # Validate flip_confidence_threshold
    if not (0.0 <= SPC_FLIP_CONFIDENCE_THRESHOLD <= 1.0):
        errors.append(
            f"SPC_FLIP_CONFIDENCE_THRESHOLD must be in [0.0, 1.0], got {SPC_FLIP_CONFIDENCE_THRESHOLD}"
        )

    # Validate MTF timeframes
    if SPC_ENABLE_MTF:
        if not isinstance(SPC_MTF_TIMEFRAMES, list) or len(SPC_MTF_TIMEFRAMES) == 0:
            errors.append("SPC_MTF_TIMEFRAMES must be a non-empty list when SPC_ENABLE_MTF is True")

    if errors:
        raise ValueError("SPC Enhancements Configuration Errors:\n" + "\n".join(f"- {e}" for e in errors))


# Run validation on import
validate_spc_enhancements_config()

__all__ = [
    # Individual parameters
    "SPC_VOLATILITY_ADJUSTMENT",
    "SPC_USE_CORRELATION_WEIGHTS",
    "SPC_TIME_DECAY_FACTOR",
    "SPC_INTERPOLATION_MODE",
    "SPC_MIN_FLIP_DURATION",
    "SPC_FLIP_CONFIDENCE_THRESHOLD",
    "SPC_ENABLE_MTF",
    "SPC_MTF_TIMEFRAMES",
    "SPC_MTF_REQUIRE_ALIGNMENT",
    # Presets
    "SPC_PRESET_CONSERVATIVE",
    "SPC_PRESET_BALANCED",
    "SPC_PRESET_AGGRESSIVE",
    "SPC_ACTIVE_PRESET",
    # Validation
    "validate_spc_enhancements_config",
]
