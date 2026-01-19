"""Range Oscillator heatmap color calculation.

This module provides functions to calculate heatmap colors for the Range Oscillator
indicator, porting the `getHeatColor()` logic from Pine Script.

The heatmap visualizes "heat zones" based on oscillator value histograms, where
colors indicate the frequency of oscillator values touching specific levels.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from modules.range_oscillator.config.heatmap_config import HeatmapConfig


def calculate_trend_direction(
    close: pd.Series,
    ma: pd.Series,
) -> pd.Series:
    """Calculate trend direction for heatmap coloring.

    Trend direction is determined by comparing close price to moving average:
    - 1 (bullish): close > ma
    - -1 (bearish): close < ma
    - 0 (neutral): close == ma (or persists previous value)

    The trend direction persists when close == ma, matching Pine Script behavior.

    Port of Pine Script logic (lines 114-115):
    ```
    trendDir := close > ma ? 1 : close < ma ? -1 : nz(trendDir[1])
    ```

    Optimized with vectorized operations and int8 dtype for memory efficiency.

    Args:
        close: Close price series.
        ma: Moving average series (must have same index as close).

    Returns:
        Series with trend direction values (1, -1, or 0).
    """
    if not isinstance(close, pd.Series) or not isinstance(ma, pd.Series):
        raise TypeError("close and ma must be pandas Series")
    if not close.index.equals(ma.index):
        raise ValueError("close and ma must have the same index")

    # Use int8 for memory efficiency (values are only -1, 0, 1)
    trend_dir = pd.Series(0, index=close.index, dtype="int8")

    # Vectorized comparison (aligned automatically by pandas)
    bullish_mask = close > ma
    bearish_mask = close < ma

    trend_dir[bullish_mask] = 1
    trend_dir[bearish_mask] = -1

    # For neutral cases (close == ma), persist previous value (forward fill)
    # Replace 0 with NaN, then forward fill, then fill remaining NaN with 0
    trend_dir = trend_dir.replace(0, np.nan).ffill().fillna(0).astype("int8")

    return trend_dir


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., "#ff0000").

    Returns:
        RGB tuple (r, g, b) with values 0-255.
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color string.

    Args:
        rgb: RGB tuple (r, g, b) with values 0-255.

    Returns:
        Hex color string (e.g., "#ff0000").
    """
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _interpolate_color(
    value: float,
    min_val: float,
    max_val: float,
    color_start: str,
    color_end: str,
) -> str:
    """Interpolate between two colors based on value.

    Args:
        value: Value to interpolate (should be between min_val and max_val).
        min_val: Minimum value (maps to color_start).
        max_val: Maximum value (maps to color_end).
        color_start: Start color (hex string).
        color_end: End color (hex string).

    Returns:
        Interpolated color as hex string.
    """
    if max_val == min_val:
        return color_start

    # Clamp value to range
    t = max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    rgb_start = _hex_to_rgb(color_start)
    rgb_end = _hex_to_rgb(color_end)

    # Linear interpolation
    r = int(rgb_start[0] + t * (rgb_end[0] - rgb_start[0]))
    g = int(rgb_start[1] + t * (rgb_end[1] - rgb_start[1]))
    b = int(rgb_start[2] + t * (rgb_end[2] - rgb_start[2]))

    return _rgb_to_hex((r, g, b))


def _apply_gradient_coloring(
    touch_count: int,
    heat_thresh: int,
    cold_color: str,
    hot_color: str,
) -> str:
    """Apply gradient coloring based on touch count.

    Port of Pine Script logic (line 69):
    ```
    color col = color.from_gradient(cnt, heatThresh, heatThresh + 10,
                                    color.new(coldTrendCol, 80 - cnt), hotTrendCol)
    ```

    The gradient maps:
    - heatThresh -> cold_color (with transparency 80 - heatThresh)
    - heatThresh + 10 -> hot_color (with full opacity)

    For simplicity, we interpolate between cold and hot colors without
    transparency (UI can handle opacity separately if needed).

    Args:
        touch_count: Number of touches for the level.
        heat_thresh: Minimum touches threshold.
        cold_color: Color for low touch count.
        hot_color: Color for high touch count.
        heat_thresh: Minimum threshold for "hot" zones.

    Returns:
        Color as hex string.
    """
    min_touches = heat_thresh
    max_touches = heat_thresh + 10

    # Clamp touch_count to gradient range
    clamped_count = max(min_touches, min(max_touches, touch_count))

    return _interpolate_color(clamped_count, min_touches, max_touches, cold_color, hot_color)


def calculate_heat_colors(
    oscillator: pd.Series,
    trend_direction: pd.Series,
    config: Optional[HeatmapConfig] = None,
) -> pd.Series:
    """Calculate heatmap colors for oscillator values.

    This function ports the Pine Script `getHeatColor()` logic (lines 35-89).
    It divides the oscillator range into levels, counts touches per level,
    applies gradient coloring, and returns the color for the nearest level
    to each oscillator value.

    Args:
        oscillator: Oscillator values series.
        trend_direction: Trend direction series (1=bullish, -1=bearish, 0=neutral).
        config: Heatmap configuration. If None, uses default config.

    Returns:
        Series of hex color strings, one for each oscillator value.
        NaN values in oscillator result in transition_color.
    """
    if config is None:
        config = HeatmapConfig()

    if not isinstance(oscillator, pd.Series) or not isinstance(trend_direction, pd.Series):
        raise TypeError("oscillator and trend_direction must be pandas Series")
    if not oscillator.index.equals(trend_direction.index):
        raise ValueError("oscillator and trend_direction must have the same index")
    if len(oscillator) == 0:
        return pd.Series(dtype="object", index=oscillator.index)

    # Initialize result series with transition color
    heat_colors = pd.Series(config.transition_color, index=oscillator.index, dtype="object")

    lookback = min(config.lookback_bars, len(oscillator))

    # Pre-compute trend flips (no_color_on_flip in Pine Script) - vectorized
    trend_flip = trend_direction != trend_direction.shift(1, fill_value=trend_direction.iloc[0] if len(trend_direction) > 0 else 0)
    
    # Convert oscillator to numpy array for faster indexing
    oscillator_array = oscillator.values
    trend_direction_array = trend_direction.values
    trend_flip_array = trend_flip.values

    for i in range(len(oscillator)):
        # If trend flipped at this bar, use transition color (blue)
        if i > 0 and trend_flip_array[i]:
            continue  # Already set to transition_color

        if pd.isna(oscillator_array[i]):
            continue

        # Get lookback window (last lookback bars up to current bar)
        start_idx = max(0, i - lookback + 1)
        window_array = oscillator_array[start_idx : i + 1]

        # Remove NaN values (vectorized)
        valid_mask = ~np.isnan(window_array)
        valid_window_array = window_array[valid_mask]

        if len(valid_window_array) == 0:
            continue

        # Calculate range (vectorized)
        hi = np.max(valid_window_array)
        lo = np.min(valid_window_array)
        rng = hi - lo

        if rng <= 0:
            continue

        # Calculate step size
        step = rng / config.levels_inp

        # Determine trend colors
        trend_dir = trend_direction_array[i]
        if trend_dir == 1:  # Bullish
            cold_color = config.weak_bullish_color
            hot_color = config.strong_bullish_color
        elif trend_dir == -1:  # Bearish
            cold_color = config.weak_bearish_color
            hot_color = config.strong_bearish_color
        else:  # Neutral
            cold_color = config.transition_color
            hot_color = config.transition_color

        # Calculate levels and touch counts (vectorized)
        # valid_window_array is already computed above

        # Pre-calculate all level values
        if config.point_mode:
            level_values = np.array([lo + step * (idx + 0.5) for idx in range(config.levels_inp)])
        else:
            level_values = np.array([lo + step * idx for idx in range(config.levels_inp)])

        # Vectorized touch counting for all levels at once
        if config.point_mode:
            # Point mode: check if values are within step/2 of each level
            # Shape: (n_levels, n_values)
            distances = np.abs(valid_window_array[np.newaxis, :] - level_values[:, np.newaxis])
            touches = distances < (step / 2)
        else:
            # Range mode: same as point mode for oscillator
            distances = np.abs(valid_window_array[np.newaxis, :] - level_values[:, np.newaxis])
            touches = distances < (step / 2)

        # Count touches per level (sum along axis 1)
        touch_counts = np.sum(touches, axis=1)

        # Apply gradient coloring (vectorized where possible)
        level_colors = []
        for level_idx, cnt in enumerate(touch_counts):
            if cnt >= config.heat_thresh:
                col = _apply_gradient_coloring(cnt, config.heat_thresh, cold_color, hot_color)
            else:
                col = cold_color
            level_colors.append(col)

        # Find nearest level to current oscillator value (vectorized)
        current_val = oscillator_array[i]
        distances_to_levels = np.abs(level_values - current_val)
        nearest_idx = np.argmin(distances_to_levels)
        heat_colors.iloc[i] = level_colors[nearest_idx]

    return heat_colors


__all__ = [
    "calculate_trend_direction",
    "calculate_heat_colors",
]
