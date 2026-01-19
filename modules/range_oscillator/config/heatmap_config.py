"""Configuration for Range Oscillator heatmap visualization.

This module provides configuration for the heatmap color calculation
function, which visualizes "heat zones" based on oscillator value histograms.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HeatmapConfig:
    """Configuration for Range Oscillator heatmap calculation.

    This configuration matches the Pine Script heatmap parameters from
    the original Range Oscillator indicator (Zeiierman).

    Attributes:
        levels_inp: Number of horizontal levels (bands) used in the heatmap.
            More levels give finer granularity but may introduce noise.
            Default: 2, Range: 2-100.
        heat_thresh: Minimum number of touches per level to consider it 'hot'.
            Lower values make the heatmap more reactive.
            Default: 1, Minimum: 1.
        lookback_bars: Number of bars to look back for touch counting.
            Default: 100 (matches Pine Script).
        point_mode: Whether to use point mode for level calculation.
            In point mode, levels are calculated at midpoints between boundaries.
            Default: True (matches Pine Script).
        weak_bullish_color: Color for weak bullish zones (pressure zones in uptrends).
            Default: "#008000" (green).
        strong_bullish_color: Color for strong bullish zones (less resistance in uptrends).
            Default: "#09ff00" (bright green).
        weak_bearish_color: Color for weak bearish zones (pressure zones in downtrends).
            Default: "#800000" (maroon).
        strong_bearish_color: Color for strong bearish zones (less resistance in downtrends).
            Default: "#ff0000" (red).
        transition_color: Color used during trend transitions or when no valid
            heatmap color is available.
            Default: "#0000ff" (blue).
    """

    levels_inp: int = 2
    heat_thresh: int = 1
    lookback_bars: int = 100
    point_mode: bool = True
    weak_bullish_color: str = "#008000"  # green
    strong_bullish_color: str = "#09ff00"  # bright green
    weak_bearish_color: str = "#800000"  # maroon
    strong_bearish_color: str = "#ff0000"  # red
    transition_color: str = "#0000ff"  # blue

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.levels_inp < 2 or self.levels_inp > 100:
            raise ValueError(f"levels_inp must be between 2 and 100, got {self.levels_inp}")
        if self.heat_thresh < 1:
            raise ValueError(f"heat_thresh must be >= 1, got {self.heat_thresh}")
        if self.lookback_bars < 1:
            raise ValueError(f"lookback_bars must be >= 1, got {self.lookback_bars}")

        # Validate color format (basic hex check)
        color_fields = [
            "weak_bullish_color",
            "strong_bullish_color",
            "weak_bearish_color",
            "strong_bearish_color",
            "transition_color",
        ]
        for field in color_fields:
            color = getattr(self, field)
            if not isinstance(color, str) or not color.startswith("#"):
                raise ValueError(f"{field} must be a hex color string starting with '#', got {color}")
            if len(color) != 7:  # #RRGGBB
                raise ValueError(f"{field} must be a 7-character hex color (#RRGGBB), got {color}")


__all__ = [
    "HeatmapConfig",
]
