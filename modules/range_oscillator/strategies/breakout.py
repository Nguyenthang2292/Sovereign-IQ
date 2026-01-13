"""
Range Oscillator Strategy 6: Range Breakouts.

This module provides the range breakout signal generation strategy.
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from modules.common.indicators.volatility import calculate_atr_range
from modules.common.utils import log_analysis, log_debug
from modules.range_oscillator.utils.oscillator_data import get_oscillator_data


def _calculate_dynamic_exhaustion_threshold(
    range_atr: pd.Series,
    base_exhaustion_threshold: float,
    exhaustion_atr_multiplier: float,
    exhaustion_atr_period: int,
) -> pd.Series:
    """
    Calculate dynamic exhaustion threshold based on ATR.

    Formula: exhaustion_threshold = exhaustion_atr_multiplier * (range_atr / base_range_atr) * base_exhaustion_threshold

    Args:
        range_atr: Range ATR series (ATR * mult)
        base_exhaustion_threshold: Base threshold value
        exhaustion_atr_multiplier: Multiplier for ATR ratio
        exhaustion_atr_period: Period to calculate base_range_atr (mean of first N periods)

    Returns:
        Series with dynamic exhaustion threshold values
    """
    if len(range_atr) == 0:
        return pd.Series([base_exhaustion_threshold], index=range_atr.index, dtype="float64")

    # Calculate base_range_atr as mean of first exhaustion_atr_period periods
    # Use minimum of available data and requested period
    period_to_use = min(exhaustion_atr_period, len(range_atr))

    if period_to_use == 0:
        # Fallback to static threshold if no data
        return pd.Series([base_exhaustion_threshold] * len(range_atr), index=range_atr.index, dtype="float64")

    # Calculate base_range_atr from first N periods (use mean)
    base_range_atr = range_atr.iloc[:period_to_use].mean()

    # Handle edge cases: zero or NaN base_range_atr
    if pd.isna(base_range_atr) or base_range_atr <= 0:
        # Fallback: use current range_atr value or a small positive value
        base_range_atr = range_atr.dropna()
        if len(base_range_atr) > 0:
            base_range_atr = max(base_range_atr.min(), 1.0)  # At least 1.0 to avoid division issues
        else:
            base_range_atr = 1.0

    # Calculate dynamic threshold: multiplier * (current_range_atr / base_range_atr) * base_threshold
    # Handle NaN and zero values in range_atr
    range_atr_safe = range_atr.fillna(base_range_atr)  # Fill NaN with base value
    range_atr_safe = range_atr_safe.replace(0, base_range_atr)  # Replace 0 with base value

    # Calculate ratio
    atr_ratio = range_atr_safe / base_range_atr

    # Calculate dynamic threshold
    dynamic_threshold = exhaustion_atr_multiplier * atr_ratio * base_exhaustion_threshold

    # Ensure threshold is non-negative
    dynamic_threshold = dynamic_threshold.clip(lower=0.0)

    return dynamic_threshold


def generate_signals_breakout_strategy(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    *,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
    upper_threshold: float = 100.0,
    lower_threshold: float = -100.0,
    require_confirmation: bool = True,
    confirmation_bars: int = 2,
    detect_exhaustion: bool = True,
    exhaustion_threshold: float = 150.0,
    use_dynamic_exhaustion: bool = False,
    exhaustion_atr_multiplier: float = 1.0,
    base_exhaustion_threshold: float = 150.0,
    exhaustion_atr_period: int = 50,
    enable_debug: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate trading signals based on Range Oscillator Strategy 6: Range Breakouts.

    Strategy Logic:
    ---------------
    This strategy focuses on range breakouts as shown in TradingView charts:
    - LONG Signal: Oscillator breaks above +100 (breakout potential)
    - SHORT Signal: Oscillator breaks below -100 (breakout potential)
    - Exhaustion Detection: Oscillator > +150 or < -150 (potential reversal)

    Key Concept:
    "When the oscillator line breaks above +100 or below -100, the price is
    exceeding its normal volatility range, often signaling breakout potential
    or exhaustion extremes."

    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional, if provided with ma and range_atr)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50, ignored if oscillator provided)
        mult: Range width multiplier (default: 2.0, ignored if oscillator provided)
        upper_threshold: Upper breakout threshold (default: 100.0)
        lower_threshold: Lower breakout threshold (default: -100.0)
        require_confirmation: If True, require confirmation bars (default: True)
        confirmation_bars: Number of bars to confirm breakout (default: 2)
        detect_exhaustion: If True, detect exhaustion extremes (default: True)
        exhaustion_threshold: Threshold for exhaustion detection
            (default: 150.0, used when use_dynamic_exhaustion=False)
        use_dynamic_exhaustion: If True, use ATR-based dynamic exhaustion threshold (default: False)
        exhaustion_atr_multiplier: Multiplier for ATR ratio in dynamic threshold calculation (default: 1.0)
        base_exhaustion_threshold: Base threshold value for dynamic calculation (default: 150.0)
        exhaustion_atr_period: Period to calculate base_range_atr for dynamic threshold (default: 50)
        enable_debug: If True, enable debug logging (default: False)

    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Validate parameters
    if require_confirmation:
        if confirmation_bars <= 0:
            raise ValueError(f"confirmation_bars must be > 0, got {confirmation_bars}")

    if detect_exhaustion:
        if not use_dynamic_exhaustion:
            if exhaustion_threshold < 0:
                raise ValueError(f"exhaustion_threshold must be >= 0, got {exhaustion_threshold}")
        else:
            if exhaustion_atr_multiplier <= 0:
                raise ValueError(f"exhaustion_atr_multiplier must be > 0, got {exhaustion_atr_multiplier}")
            if base_exhaustion_threshold < 0:
                raise ValueError(f"base_exhaustion_threshold must be >= 0, got {base_exhaustion_threshold}")
            if exhaustion_atr_period <= 0:
                raise ValueError(f"exhaustion_atr_period must be > 0, got {exhaustion_atr_period}")

    if upper_threshold <= lower_threshold:
        raise ValueError(f"upper_threshold ({upper_threshold}) must be > lower_threshold ({lower_threshold})")

    # Enable debug logging if requested
    debug_enabled = enable_debug or os.environ.get("RANGE_OSCILLATOR_DEBUG", "false").lower() == "true"

    if debug_enabled:
        log_analysis("[Strategy6] Starting breakout signal generation")
        log_debug(
            f"[Strategy6] Parameters: upper_threshold={upper_threshold}, "
            f"lower_threshold={lower_threshold}, require_confirmation={require_confirmation}, "
            f"confirmation_bars={confirmation_bars}, detect_exhaustion={detect_exhaustion}, "
            f"exhaustion_threshold={exhaustion_threshold}"
        )

    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = get_oscillator_data(
        high=high, low=low, close=close, oscillator=oscillator, ma=ma, range_atr=range_atr, length=length, mult=mult
    )

    # If dynamic exhaustion is enabled, ensure range_atr has valid data
    # Note: After get_oscillator_data() succeeds, range_atr cannot be None or empty,
    # but we check for all-NaN case as a safety measure
    if detect_exhaustion and use_dynamic_exhaustion and not range_atr.notna().any():
        if high is not None and low is not None and close is not None:
            # Calculate ATR range using common indicators
            range_atr = calculate_atr_range(
                high=high,
                low=low,
                close=close,
                mult=mult,
            )
        else:
            # Fallback to static threshold if cannot calculate ATR
            if debug_enabled:
                log_debug("[Strategy6] Cannot calculate ATR for dynamic exhaustion, falling back to static threshold")
            use_dynamic_exhaustion = False

    if debug_enabled:
        log_debug(f"[Strategy6] Data shape: oscillator={len(oscillator)}")
        if detect_exhaustion and use_dynamic_exhaustion:
            log_debug(
                f"[Strategy6] Dynamic exhaustion enabled: multiplier={exhaustion_atr_multiplier}, "
                f"base_threshold={base_exhaustion_threshold}, period={exhaustion_atr_period}"
            )

    # Vectorized exhaustion detection
    in_exhaustion_zone = pd.Series(False, index=oscillator.index)
    if detect_exhaustion:
        if use_dynamic_exhaustion:
            # Calculate dynamic threshold
            dynamic_threshold = _calculate_dynamic_exhaustion_threshold(
                range_atr=range_atr,
                base_exhaustion_threshold=base_exhaustion_threshold,
                exhaustion_atr_multiplier=exhaustion_atr_multiplier,
                exhaustion_atr_period=exhaustion_atr_period,
            )

            # Ensure index alignment
            if not dynamic_threshold.index.equals(oscillator.index):
                dynamic_threshold = dynamic_threshold.reindex(oscillator.index, method="ffill").fillna(
                    base_exhaustion_threshold
                )

            in_exhaustion_zone = oscillator.abs() >= dynamic_threshold

            if debug_enabled:
                threshold_min = float(dynamic_threshold.min())
                threshold_max = float(dynamic_threshold.max())
                threshold_mean = float(dynamic_threshold.mean())
                log_debug(
                    f"[Strategy6] Dynamic threshold stats: min={threshold_min:.2f}, "
                    f"max={threshold_max:.2f}, mean={threshold_mean:.2f}, "
                    f"static={exhaustion_threshold}"
                )
        else:
            # Use static threshold (backward compatibility)
            in_exhaustion_zone = oscillator.abs() >= exhaustion_threshold

    # Detect breakouts using shift
    prev_osc = oscillator.shift(1)
    breakout_above = (prev_osc <= upper_threshold) & (oscillator > upper_threshold)
    breakout_below = (prev_osc >= lower_threshold) & (oscillator < lower_threshold)

    # Initialize signals
    signals = pd.Series(0, index=oscillator.index, dtype="int8")

    if require_confirmation:
        # Count consecutive bars above/below threshold
        above_threshold = (oscillator > upper_threshold).astype(int)
        below_threshold = (oscillator < lower_threshold).astype(int)

        # Use groupby to count consecutive bars (optimized: sort=False for large datasets)
        above_groups = (above_threshold != above_threshold.shift()).cumsum()
        above_counts = above_threshold.groupby(above_groups, sort=False).cumsum()

        below_groups = (below_threshold != below_threshold.shift()).cumsum()
        below_counts = below_threshold.groupby(below_groups, sort=False).cumsum()

        # FIXED: Avoid look-ahead bias by emitting signals AFTER confirmation period
        # Logic: Signal is only emitted AFTER confirmation_bars bars have passed since breakout
        #
        # Example with confirmation_bars=2:
        # - Breakout occurs at t=5 (oscillator breaks above upper_threshold)
        # - Signal is emitted at t=7 (after 2 bars of confirmation)
        # - This ensures we only trade after confirmation is complete, not at breakout time

        # Shift breakout detection forward by confirmation_bars
        # This means: if breakout happened at t, signal will be emitted at t + confirmation_bars
        # Using shift(confirmation_bars) moves the breakout signal forward in time
        breakout_above_shifted = breakout_above.shift(confirmation_bars)
        breakout_below_shifted = breakout_below.shift(confirmation_bars)

        # At current time t, check if:
        # 1. Breakout occurred at t - confirmation_bars (via shifted signal)
        # 2. Current consecutive count >= confirmation_bars (oscillator has stayed above/below for enough bars)
        # 3. We're currently above/below threshold
        #
        # Note: above_counts[t] represents consecutive bars above threshold ending at t
        # If above_counts[t] >= confirmation_bars, it means oscillator has been above threshold
        # for at least confirmation_bars consecutive bars, which confirms the breakout

        # For LONG: breakout happened confirmation_bars ago AND oscillator has stayed above threshold
        # Convert to boolean first, then fillna to avoid downcasting warning
        breakout_above_bool = breakout_above_shifted.astype(bool).fillna(False)
        above_confirmation_mask = (
            (above_counts >= confirmation_bars)
            & (above_threshold.astype(bool))
            & breakout_above_bool  # Breakout occurred confirmation_bars ago (fill NaN for beginning)
        )

        # For SHORT: breakout happened confirmation_bars ago AND oscillator has stayed below threshold
        breakout_below_bool = breakout_below_shifted.astype(bool).fillna(False)
        below_confirmation_mask = (
            (below_counts >= confirmation_bars)
            & (below_threshold.astype(bool))
            & breakout_below_bool  # Breakout occurred confirmation_bars ago (fill NaN for beginning)
        )

        # Confirmed breakouts: signal emitted AFTER confirmation period
        breakout_above_confirmed = above_confirmation_mask
        breakout_below_confirmed = below_confirmation_mask
    else:
        breakout_above_confirmed = breakout_above
        breakout_below_confirmed = breakout_below

    if debug_enabled:
        breakout_above_count = int(breakout_above.sum())
        breakout_below_count = int(breakout_below.sum())
        confirmed_above = int(breakout_above_confirmed.sum())
        confirmed_below = int(breakout_below_confirmed.sum())
        exhaustion_count = int(in_exhaustion_zone.sum()) if detect_exhaustion else 0
        log_debug(
            f"[Strategy6] Breakouts: above={breakout_above_count} (confirmed={confirmed_above}), "
            f"below={breakout_below_count} (confirmed={confirmed_below}), "
            f"exhaustion_zones={exhaustion_count}"
        )

    # Generate signals
    # LONG signals: confirmed breakout above and not in exhaustion zone
    long_condition = breakout_above_confirmed & ~in_exhaustion_zone
    # SHORT signals: confirmed breakout below and not in exhaustion zone
    short_condition = breakout_below_confirmed & ~in_exhaustion_zone

    # Handle conflict: if both conditions are met, set to NEUTRAL (0)
    # This can happen in edge cases with NaN values or data inconsistencies
    conflict_mask = long_condition & short_condition
    long_condition = long_condition & ~conflict_mask
    short_condition = short_condition & ~conflict_mask

    if debug_enabled:
        conflict_count = int(conflict_mask.sum())
        if conflict_count > 0:
            log_debug(f"[Strategy6] Conflicts detected: {conflict_count} (set to NEUTRAL)")

    signals = np.where(long_condition, 1, signals)
    signals = np.where(short_condition, -1, signals)

    # Maintain signal while oscillator stays in breakout zone (vectorized)
    above_zone = oscillator > upper_threshold
    below_zone = oscillator < lower_threshold

    # Optimized forward fill using ffill() with groupby
    signal_series = pd.Series(signals, index=oscillator.index, dtype="int8")

    # For LONG signals: forward fill 1 while in above zone
    # Create groups: each group is a continuous period in above zone
    # Optimized: use sort=False for better performance
    long_groups = (~above_zone).cumsum()
    # Keep only LONG signals (1) and forward fill within each group
    long_signals = signal_series.where(signal_series == 1, np.nan)
    long_signals = long_signals.groupby(long_groups, sort=False).ffill()
    # Reset to 0 where condition is not met
    long_signals = long_signals.where(above_zone, 0).fillna(0).astype("int8")

    # For SHORT signals: forward fill -1 while in below zone
    # Create groups: each group is a continuous period in below zone
    # Optimized: use sort=False for better performance
    short_groups = (~below_zone).cumsum()
    # Keep only SHORT signals (-1) and forward fill within each group
    short_signals = signal_series.where(signal_series == -1, np.nan)
    short_signals = short_signals.groupby(short_groups, sort=False).ffill()
    # Reset to 0 where condition is not met
    short_signals = short_signals.where(below_zone, 0).fillna(0).astype("int8")

    # Combine signals: LONG if long_signals == 1, SHORT if short_signals == -1
    # Optimized: use numpy operations directly
    signals_array = np.where(long_signals.values == 1, 1, np.where(short_signals.values == -1, -1, 0))
    signals = pd.Series(signals_array, index=oscillator.index, dtype="int8")

    # Calculate signal strength
    excess_above = np.clip((oscillator - upper_threshold) / 50.0, 0.0, 1.0)
    excess_below = np.clip((lower_threshold - oscillator) / 50.0, 0.0, 1.0)

    # Initialize signal strength
    signal_strength = np.zeros(len(oscillator), dtype=np.float64)

    # Strength for confirmed breakouts (higher strength)
    long_mask = signals == 1
    short_mask = signals == -1
    signal_strength = np.where(long_mask & breakout_above_confirmed, np.maximum(excess_above, 0.3), signal_strength)
    signal_strength = np.where(short_mask & breakout_below_confirmed, np.maximum(excess_below, 0.3), signal_strength)

    # Strength for maintained signals in zone (lower strength)
    signal_strength = np.where(
        long_mask & above_zone & ~breakout_above_confirmed, np.maximum(excess_above, 0.2), signal_strength
    )
    signal_strength = np.where(
        short_mask & below_zone & ~breakout_below_confirmed, np.maximum(excess_below, 0.2), signal_strength
    )

    signal_strength = pd.Series(signal_strength, index=oscillator.index, dtype="float64")

    # Handle NaN and exhaustion zone (optimized: combine all checks)
    valid_mask = ~(oscillator.isna() | in_exhaustion_zone)
    signals = signals.where(valid_mask, 0)
    signal_strength = signal_strength.where(valid_mask, 0.0)

    if debug_enabled:
        final_long = int((signals == 1).sum())
        final_short = int((signals == -1).sum())
        final_neutral = int((signals == 0).sum())
        avg_strength = float(signal_strength.mean())
        log_analysis(
            f"[Strategy6] Final signals: LONG={final_long}, SHORT={final_short}, "
            f"NEUTRAL={final_neutral}, avg_strength={avg_strength:.3f}"
        )

    return signals, signal_strength


__all__ = [
    "generate_signals_breakout_strategy",
]
