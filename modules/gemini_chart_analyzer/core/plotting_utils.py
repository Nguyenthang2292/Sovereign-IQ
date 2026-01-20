"""
Common plotting utilities for technical analysis charts.

This module provides reusable functions for plotting candlestick charts, indicators,
and volume, ensuring consistent styling and performance across different generators.
"""

import itertools
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle

from modules.common.ui.logging import log_warn

# Constants
CANDLE_WIDTH_RATIO = 0.8
FALLBACK_CANDLE_WIDTH = 0.01
UP_COLOR = "#00ff00"
DOWN_COLOR = "#ff0000"
DEFAULT_MA_COLORS = ["blue", "orange", "red", "purple", "cyan"]


def setup_chart_style(ax, title: str = "", ylabel: str = "Price", show_grid: bool = True):
    """Apply consistent styling to a chart axis."""
    ax.set_title(title, fontsize=14, fontweight="bold", color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.tick_params(colors="white")
    if show_grid:
        ax.grid(True, alpha=0.3, linewidth=0.5, color="gray")
    ax.set_facecolor("black")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())


def create_figure(rows: int, figsize: Tuple[int, int], height_ratios: List[int]) -> Tuple[plt.Figure, Any]:
    """Create a figure with multiple subplots and specified height ratios."""
    fig, axes = plt.subplots(rows, 1, figsize=figsize, gridspec_kw={"height_ratios": height_ratios})
    if rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else list(axes)
    return fig, axes


def calculate_candle_width(df_candle: pd.DataFrame) -> float:
    """Calculate candle width based on time intervals in data."""
    if len(df_candle) > 1:
        time_diffs = df_candle.index.to_series().diff().dt.total_seconds() / 86400
        positive_diffs = time_diffs[time_diffs > 0]
        if len(positive_diffs) > 0:
            median_diff = positive_diffs.median()
            if pd.notna(median_diff) and median_diff > 0:
                return median_diff * CANDLE_WIDTH_RATIO
    return FALLBACK_CANDLE_WIDTH


def plot_candlesticks(ax, df: pd.DataFrame, candle_width: float):
    """Plot candlestick chart using vectorized operations (fast)."""
    x_numeric = mdates.date2num(df.index)
    opens, highs, lows, closes = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    is_up = closes >= opens
    colors = np.where(is_up, UP_COLOR, DOWN_COLOR)

    body_lows, body_highs = np.minimum(opens, closes), np.maximum(opens, closes)
    body_heights = body_highs - body_lows

    has_body = body_heights > 0
    is_doji = ~has_body

    body_indices = np.where(has_body)[0]
    rectangles = [
        Rectangle((x_numeric[i] - candle_width / 2, body_lows[i]), candle_width, body_heights[i]) for i in body_indices
    ]
    if rectangles:
        body_collection = PatchCollection(
            rectangles, facecolors=colors[body_indices], edgecolors=colors[body_indices], linewidths=0.5
        )
        ax.add_collection(body_collection)

    if np.any(is_doji):
        doji_indices = np.where(is_doji)[0]
        doji_x = x_numeric[doji_indices]
        doji_y = opens[doji_indices]
        doji_segments = np.zeros((len(doji_indices), 2, 2))
        doji_segments[:, 0, 0] = doji_x - candle_width / 2
        doji_segments[:, 0, 1] = doji_y
        doji_segments[:, 1, 0] = doji_x + candle_width / 2
        doji_segments[:, 1, 1] = doji_y
        ax.add_collection(LineCollection(doji_segments, colors=colors[doji_indices], linewidths=1.5))

    wick_segments = np.zeros((len(df), 2, 2))
    wick_segments[:, 0, 0], wick_segments[:, 0, 1] = x_numeric, lows
    wick_segments[:, 1, 0], wick_segments[:, 1, 1] = x_numeric, highs
    ax.add_collection(LineCollection(wick_segments, colors=colors, linewidths=0.8, alpha=0.8))
    ax.autoscale_view()


def plot_moving_averages(ax, df: pd.DataFrame, ma_config: Dict):
    """Plot multiple moving averages on the given axis."""
    periods = ma_config.get("periods", [20, 50, 200])
    colors = ma_config.get("colors", DEFAULT_MA_COLORS)
    color_iter = itertools.cycle(colors)

    for period in periods:
        col_name = f"MA_{period}"
        if col_name in df.columns:
            ax.plot(df.index, df[col_name], label=f"MA {period}", color=next(color_iter), linewidth=1.5, alpha=0.8)


def plot_bollinger_bands(ax, df: pd.DataFrame, bb_config: Dict):
    """Plot Bollinger Bands on the given axis."""
    period = bb_config.get("period", 20)
    if f"BB_upper_{period}" in df.columns:
        ax.plot(
            df.index, df[f"BB_upper_{period}"], label="BB Upper", color="cyan", linewidth=1, alpha=0.6, linestyle="--"
        )
        ax.plot(
            df.index, df[f"BB_lower_{period}"], label="BB Lower", color="cyan", linewidth=1, alpha=0.6, linestyle="--"
        )
        ax.plot(df.index, df[f"BB_middle_{period}"], label="BB Middle", color="yellow", linewidth=1, alpha=0.4)


def plot_volume(ax, df: pd.DataFrame):
    """Plot volume bars with color matching price movement."""
    colors = np.where(df["close"] >= df["open"], UP_COLOR, DOWN_COLOR)
    ax.bar(df.index, df["volume"], color=colors, alpha=0.6)
    setup_chart_style(ax, title="", ylabel="Volume", show_grid=True)


def prepare_dataframe_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has DatetimeIndex and required OHLCV columns."""
    # Check for empty DataFrame first
    if df.empty:
        raise ValueError("Empty DataFrame, cannot create chart")

    df_plot = df.copy()

    # Check for required columns before DatetimeIndex
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df_plot.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if not isinstance(df_plot.index, pd.DatetimeIndex):
        if "timestamp" in df_plot.columns:
            df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
            df_plot.set_index("timestamp", inplace=True)
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'timestamp' column")

    return df_plot
