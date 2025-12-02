"""Range Oscillator indicator (Zeiierman).

This module implements the Range Oscillator indicator ported from Pine Script.
The indicator calculates a weighted moving average based on price deltas and
uses ATR-based range bands to create an oscillator with dynamic heatmap colors.

LUỒNG HOẠT ĐỘNG:
================

1. HELPER FUNCTIONS (Color Utilities)
   - _interpolate_color: Interpolate giữa 2 màu hex để tạo gradient

2. CORE CALCULATIONS (Tính toán cơ bản)
   - calculate_weighted_ma: Tính weighted MA dựa trên price deltas
   - calculate_atr_range: Tính ATR-based range bands
   - calculate_trend_direction: Xác định trend direction (bullish/bearish)

3. VISUALIZATION (Màu sắc và heatmap)
   - get_heatmap_color: Tính màu heatmap dựa trên levels và touches

4. MAIN FUNCTION (Hàm chính)
   - calculate_range_oscillator: Orchestrates toàn bộ quá trình tính toán

CHI TIẾT LUỒNG HOẠT ĐỘNG:
==========================

Bước 1: Tính Weighted Moving Average
  - Với mỗi bar, tính delta = |close[i] - close[i+1]|
  - Weight w = delta / close[i+1]
  - Weighted MA = Σ(close[i] * w) / Σ(w)
  - Mục đích: Nhấn mạnh các bar có biến động lớn hơn

Bước 2: Tính ATR Range
  - Tính ATR với length 2000 (fallback 200 nếu không đủ data)
  - Range ATR = ATR * multiplier (default 2.0)
  - Mục đích: Xác định độ rộng của range bands

Bước 3: Xác định Trend Direction
  - So sánh close với MA:
    * close > MA → trend = 1 (bullish)
    * close < MA → trend = -1 (bearish)
    * close == MA → giữ giá trị trước đó
  - Mục đích: Xác định bias để chọn màu heatmap phù hợp

Bước 4: Tính Oscillator Value
  - Oscillator = 100 * (close - MA) / RangeATR
  - Giá trị từ -100 đến +100:
    * +100: Price ở upper bound của range
    * 0: Price ở equilibrium (MA)
    * -100: Price ở lower bound của range

Bước 5: Tính Heatmap Colors
  - Chia last 100 oscillator values thành N levels
  - Đếm số lần mỗi level được "touch"
  - Gradient màu dựa trên số touches:
    * < heat_thresh → cold color (weak)
    * >= heat_thresh + 10 → hot color (strong)
    * Giữa → gradient interpolation
  - Tìm level gần nhất với giá trị hiện tại và trả về màu

Bước 6: Xác định Final Color
  - Breakout lên trên (close > MA + RangeATR) → strong bullish color
  - Breakout xuống dưới (close < MA - RangeATR) → strong bearish color
  - Trend flip (trend_dir thay đổi) → transition color
  - Còn lại → heatmap color

Original Pine Script:
    https://creativecommons.org/licenses/by-nc-sa/4.0/
    © Zeiierman
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta


# ============================================================================
# 1. HELPER FUNCTIONS - Color Utilities
# ============================================================================


def _interpolate_color(color1: str, color2: str, ratio: float) -> str:
    """Interpolate between two hex colors to create gradient.

    Helper function used by get_heatmap_color to create smooth color transitions
    between cold and hot zones based on touch count.

    Args:
        color1: First hex color (e.g., "#008000").
        color2: Second hex color (e.g., "#09ff00").
        ratio: Interpolation ratio (0.0 = color1, 1.0 = color2).

    Returns:
        Interpolated hex color string.
    """
    ratio = max(0.0, min(1.0, ratio))

    def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
        hex_str = hex_str.lstrip("#")
        return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))

    def rgb_to_hex(r: int, g: int, b: int) -> str:
        return f"#{r:02x}{g:02x}{b:02x}"

    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)

    r = int(rgb1[0] * (1 - ratio) + rgb2[0] * ratio)
    g = int(rgb1[1] * (1 - ratio) + rgb2[1] * ratio)
    b = int(rgb1[2] * (1 - ratio) + rgb2[2] * ratio)

    return rgb_to_hex(r, g, b)


# ============================================================================
# 2. CORE CALCULATIONS - Basic Metrics
# ============================================================================


def calculate_weighted_ma(
    close: pd.Series,
    length: int = 50,
) -> pd.Series:
    """Calculate weighted moving average based on price deltas.

    This function calculates a weighted moving average where larger price
    movements receive higher weights. This emphasizes recent volatility and
    creates a more responsive equilibrium line compared to simple MA.

    Port of Pine Script logic:
        sumWeightedClose = 0.0
        sumWeights = 0.0
        for i = 0 to length - 1 by 1
            delta = math.abs(close[i] - close[i + 1])
            w = delta / close[i + 1]
            sumWeightedClose := sumWeightedClose + close[i] * w
            sumWeights := sumWeights + w
        ma = sumWeights != 0 ? sumWeightedClose / sumWeights : na

    Args:
        close: Close price series.
        length: Number of bars to use for calculation (default: 50).

    Returns:
        Series containing weighted moving average values.
        First `length` values are NaN.
    """
    if len(close) < length + 1:
        return pd.Series(np.nan, index=close.index, dtype="float64")

    ma_values = []
    for i in range(len(close)):
        if i < length:
            ma_values.append(np.nan)
            continue

        sum_weighted_close = 0.0
        sum_weights = 0.0

        for j in range(length):
            idx = i - j
            prev_idx = idx - 1
            if prev_idx < 0:
                break

            delta = abs(close.iloc[idx] - close.iloc[prev_idx])
            w = delta / close.iloc[prev_idx] if close.iloc[prev_idx] != 0 else 0.0

            sum_weighted_close += close.iloc[idx] * w
            sum_weights += w

        ma_value = sum_weighted_close / sum_weights if sum_weights != 0 else np.nan
        ma_values.append(ma_value)

    return pd.Series(ma_values, index=close.index, dtype="float64")


def calculate_atr_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    mult: float = 2.0,
    atr_length_primary: int = 2000,
    atr_length_fallback: int = 200,
) -> pd.Series:
    """Calculate ATR-based range bands.

    Calculates the Average True Range (ATR) and multiplies it by a factor
    to create dynamic range bands. These bands adapt to market volatility,
    expanding during volatile periods and contracting during quiet periods.

    Port of Pine Script logic:
        atrRaw = nz(ta.atr(2000), ta.atr(200))
        rangeATR = atrRaw * mult

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        mult: Multiplier for ATR (default: 2.0).
        atr_length_primary: Primary ATR length (default: 2000).
        atr_length_fallback: Fallback ATR length if primary fails (default: 200).

    Returns:
        Series containing ATR-based range values.
    """
    # Try primary ATR length first
    atr_raw = ta.atr(high, low, close, length=atr_length_primary)
    if atr_raw is None or atr_raw.isna().all():
        # Fallback to shorter ATR
        atr_raw = ta.atr(high, low, close, length=atr_length_fallback)

    if atr_raw is None:
        # If both fail, return NaN series
        return pd.Series(np.nan, index=close.index, dtype="float64")

    # Fill NaN values forward, then backward
    atr_raw = atr_raw.ffill().bfill()
    if atr_raw.isna().all():
        # If still all NaN, use a default value
        atr_raw = pd.Series(close * 0.01, index=close.index)

    range_atr = atr_raw * mult
    return range_atr


def calculate_trend_direction(
    close: pd.Series,
    ma: pd.Series,
) -> pd.Series:
    """Calculate trend direction based on close vs weighted MA.

    Determines whether the current price is above or below the weighted MA,
    indicating bullish or bearish bias. This is used to select appropriate
    heatmap colors (bullish colors vs bearish colors).

    Port of Pine Script logic:
        var int trendDir = 0
        trendDir := close > ma ? 1 : close < ma ? -1 : nz(trendDir[1])

    Args:
        close: Close price series.
        ma: Moving average series (typically from calculate_weighted_ma).

    Returns:
        Series with trend direction:
        - 1: Bullish (close > MA)
        - -1: Bearish (close < MA)
        - 0: Neutral (uses previous value if close == MA)
    """
    trend_dir = pd.Series(0, index=close.index, dtype="int8")

    for i in range(len(close)):
        if pd.isna(close.iloc[i]) or pd.isna(ma.iloc[i]):
            # Use previous value if available
            if i > 0:
                trend_dir.iloc[i] = trend_dir.iloc[i - 1]
            continue

        if close.iloc[i] > ma.iloc[i]:
            trend_dir.iloc[i] = 1
        elif close.iloc[i] < ma.iloc[i]:
            trend_dir.iloc[i] = -1
        else:
            # Use previous value
            if i > 0:
                trend_dir.iloc[i] = trend_dir.iloc[i - 1]

    return trend_dir


# ============================================================================
# 3. VISUALIZATION - Heatmap Colors
# ============================================================================


def get_heatmap_color(
    val: float,
    trend_dir: int,
    source_series: pd.Series,
    levels_inp: int = 2,
    heat_thresh: int = 1,
    weak_bullish_color: str = "#008000",  # green
    strong_bullish_color: str = "#09ff00",  # bright green
    weak_bearish_color: str = "#800000",  # maroon
    strong_bearish_color: str = "#ff0000",  # red
    point_mode: bool = True,
) -> Optional[str]:
    """Calculate heatmap color for a given oscillator value.

    This function creates a heatmap visualization by:
    1. Dividing the last 100 oscillator values into N levels
    2. Counting how many times each level was "touched"
    3. Assigning colors based on touch count (cold → hot gradient)
    4. Finding the closest level to the current value and returning its color

    The heatmap helps visualize where price has been trading most frequently
    within the oscillator range, highlighting zones of high/low activity.

    Port of Pine Script getHeatColor function.

    Args:
        val: Current oscillator value to get color for.
        trend_dir: Trend direction (1 for bullish, -1 for bearish).
        source_series: Source series for calculating levels (last 100 values used).
        levels_inp: Number of heat levels (default: 2).
        heat_thresh: Minimum touches per level (default: 1).
        weak_bullish_color: Hex color for weak bullish zones.
        strong_bullish_color: Hex color for strong bullish zones.
        weak_bearish_color: Hex color for weak bearish zones.
        strong_bearish_color: Hex color for strong bearish zones.
        point_mode: If True, uses point mode (default: True).

    Returns:
        Hex color string or None if no valid color found.
    """
    if len(source_series) < 100:
        return None

    # Get last 100 values
    recent = source_series.iloc[-100:].dropna()
    if len(recent) < 2:
        return None

    hi = recent.max()
    lo = recent.min()
    rng = hi - lo

    if rng <= 0:
        return None

    step = rng / levels_inp

    # Determine trend colors
    # For bullish trend: cold = weak bullish, hot = strong bullish
    # For bearish trend: cold = weak bearish, hot = strong bearish
    if trend_dir == 1:
        cold_trend_col = weak_bullish_color
        hot_trend_col = strong_bullish_color
    else:
        cold_trend_col = weak_bearish_color
        hot_trend_col = strong_bearish_color

    # Calculate levels and counts
    level_vals = []
    level_colors = []

    for i in range(levels_inp):
        if point_mode:
            lvl = lo + step * (i + 0.5)
        else:
            lvl = lo + step * i

        # Count touches
        cnt = 0
        for j in range(len(recent)):
            if point_mode:
                touch = (
                    recent.iloc[j] >= lvl - step / 2
                    and recent.iloc[j] < lvl + step / 2
                )
            else:
                # For range mode, would need high/low series
                # Simplified to point mode logic
                touch = recent.iloc[j] >= lvl - step / 2 and recent.iloc[j] < lvl + step / 2

            if touch:
                cnt += 1

        # Calculate color gradient
        if cnt < heat_thresh:
            col = cold_trend_col
        elif cnt >= heat_thresh + 10:
            col = hot_trend_col
        else:
            # Gradient between cold and hot
            ratio = (cnt - heat_thresh) / 10.0
            col = _interpolate_color(cold_trend_col, hot_trend_col, ratio)

        level_vals.append(lvl)
        level_colors.append(col)

    # Find closest level
    min_d = float("inf")
    best = None

    for k in range(len(level_vals)):
        lvl = level_vals[k]
        d = abs(val - lvl)
        if d < min_d:
            min_d = d
            best = level_colors[k]

    return best


# ============================================================================
# 4. MAIN FUNCTION - Range Oscillator Calculation
# ============================================================================


def calculate_range_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    length: int = 50,
    mult: float = 2.0,
    levels_inp: int = 2,
    heat_thresh: int = 1,
    strong_bullish_color: str = "#09ff00",
    strong_bearish_color: str = "#ff0000",
    weak_bearish_color: str = "#800000",
    weak_bullish_color: str = "#008000",
    transition_color: str = "#0000ff",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Range Oscillator indicator.

    This is the main function that orchestrates the entire Range Oscillator
    calculation process. It combines weighted MA, ATR range, trend direction,
    and heatmap colors to produce a comprehensive oscillator indicator.

    LUỒNG TÍNH TOÁN:
    ----------------
    1. Tính Weighted MA từ close prices
    2. Tính ATR Range từ high/low/close
    3. Xác định Trend Direction (bullish/bearish)
    4. Với mỗi bar:
       a. Tính Oscillator = 100 * (close - MA) / RangeATR
       b. Kiểm tra breakouts (upper/lower bounds)
       c. Tính heatmap color dựa trên historical touches
       d. Xác định final color (breakout > heatmap > transition)

    Port of Pine Script Range Oscillator (Zeiierman).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        length: Minimum range length (default: 50).
        mult: Range width multiplier (default: 2.0).
        levels_inp: Number of heat levels (default: 2).
        heat_thresh: Minimum touches per level (default: 1).
        strong_bullish_color: Hex color for strong bullish zones.
        strong_bearish_color: Hex color for strong bearish zones.
        weak_bearish_color: Hex color for weak bearish zones.
        weak_bullish_color: Hex color for weak bullish zones.
        transition_color: Hex color for transitions.

    Returns:
        Tuple containing:
        - oscillator: Oscillator values (ranges from -100 to +100)
        - oscillator_color: Hex color strings for each oscillator value
        - ma: Weighted moving average
        - range_atr: ATR-based range
    """
    # Step 1: Calculate weighted MA
    ma = calculate_weighted_ma(close, length=length)

    # Step 2: Calculate ATR range
    range_atr = calculate_atr_range(high, low, close, mult=mult)

    # Step 3: Calculate trend direction
    trend_dir = calculate_trend_direction(close, ma)

    # Step 4: Calculate oscillator and colors
    oscillator = pd.Series(np.nan, index=close.index, dtype="float64")
    oscillator_color = pd.Series(None, index=close.index, dtype="object")

    prev_trend_dir = 0

    for i in range(len(close)):
        if pd.isna(range_atr.iloc[i]) or range_atr.iloc[i] == 0:
            continue

        if pd.isna(ma.iloc[i]):
            continue

        # Step 4a: Calculate oscillator value
        osc_value = 100 * (close.iloc[i] - ma.iloc[i]) / range_atr.iloc[i]
        oscillator.iloc[i] = osc_value

        # Step 4b: Determine color
        current_trend_dir = trend_dir.iloc[i]
        no_color_on_flip = current_trend_dir != prev_trend_dir

        # Step 4c: Check for breakouts
        break_up = close.iloc[i] > ma.iloc[i] + range_atr.iloc[i]
        break_dn = close.iloc[i] < ma.iloc[i] - range_atr.iloc[i]

        if break_up:
            # Price broke above upper bound → strong bullish
            osc_color = strong_bullish_color
        elif break_dn:
            # Price broke below lower bound → strong bearish
            osc_color = strong_bearish_color
        else:
            # Step 4d: Get heatmap color
            heat_color = get_heatmap_color(
                osc_value,
                current_trend_dir,
                oscillator.iloc[: i + 1],
                levels_inp=levels_inp,
                heat_thresh=heat_thresh,
                weak_bullish_color=weak_bullish_color,
                strong_bullish_color=strong_bullish_color,
                weak_bearish_color=weak_bearish_color,
                strong_bearish_color=strong_bearish_color,
                point_mode=True,
            )

            # Step 4e: Determine final color
            if heat_color is None or no_color_on_flip:
                # Trend flip or invalid heatmap → transition color
                osc_color = transition_color
            else:
                # Use heatmap color
                osc_color = heat_color

        oscillator_color.iloc[i] = osc_color
        prev_trend_dir = current_trend_dir

    return oscillator, oscillator_color, ma, range_atr


__all__ = [
    "calculate_weighted_ma",
    "calculate_atr_range",
    "calculate_trend_direction",
    "get_heatmap_color",
    "calculate_range_oscillator",
]
