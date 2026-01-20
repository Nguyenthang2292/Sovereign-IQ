"""
Chart Generator for creating technical analysis charts with indicators.

Creates candlestick charts with indicators such as MA, RSI, Volume, etc.
"""

import os
import matplotlib
import gc
import itertools
import traceback
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Use non-interactive backend to avoid GUI overhead and memory leaks
matplotlib.use("Agg")  # Must be set before importing pyplot

from modules.common.indicators import (
    calculate_bollinger_bands_series,
    calculate_ma_series,
    calculate_macd_series,
    calculate_rsi_series,
)
from modules.common.ui.logging import log_success, log_warn
from modules.gemini_chart_analyzer.core.plotting_utils import (
    calculate_candle_width,
    create_figure,
    plot_bollinger_bands,
    plot_candlesticks,
    plot_moving_averages,
    plot_volume,
    prepare_dataframe_for_plotting,
    setup_chart_style,
)
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_charts_dir


class ChartGenerator:
    """Generate technical charts with indicators."""

    def __init__(self, figsize: Tuple[int, int] = (16, 10), style: str = "dark_background", dpi: int = 150):
        """Initialize ChartGenerator."""
        self.figsize = figsize
        self.style = style
        self.dpi = dpi

    def create_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        indicators: Optional[Dict[str, Dict]] = None,
        output_path: Optional[str] = None,
        show_volume: bool = True,
    ) -> str:
        """Create a candlestick chart with indicators and save to a file."""
        # Prepare data using common utility
        df = prepare_dataframe_for_plotting(df)

        # Calculate indicators
        indicators = indicators or {}
        df = self._add_indicators(df, indicators)

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_symbol = symbol.replace("/", "_").replace(":", "_")
            charts_dir = get_charts_dir()
            os.makedirs(str(charts_dir), exist_ok=True)
            output_path = os.path.join(str(charts_dir), f"{safe_symbol}_{timeframe}_{timestamp}.png")

        # Ensure directory exists
        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Plot chart within style context
        with plt.style.context([self.style]):
            fig, axes, total_rows = self._create_subplots(indicators, show_volume)

            # Plot candlesticks
            self._plot_candlesticks(axes[0], df, symbol, timeframe, ax_index=0, total_rows=total_rows)

            # Plot price indicators (MA, BB)
            self._plot_price_indicators(axes[0], df, indicators)

            # Plot volume
            current_ax_idx = 1
            if show_volume and "volume" in df.columns:
                self._plot_volume(axes[current_ax_idx], df, ax_index=current_ax_idx, total_rows=total_rows)
                current_ax_idx += 1

            # Plot separate indicators (RSI, MACD)
            self._plot_separate_indicators(axes[current_ax_idx:], df, indicators, current_ax_idx, total_rows)

            plt.tight_layout()
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight", facecolor="black")

        plt.close(fig)
        gc.collect()

        log_success(f"Chart saved: {output_path}")
        return output_path

    def _create_subplots(self, indicators: Optional[Dict], show_volume: bool) -> Tuple:
        """Create subplots using common utility."""
        separate_indicators = ["RSI", "MACD"]
        if indicators is None:
            indicators = {}
        indicator_count = sum(1 for ind in separate_indicators if ind in indicators)

        rows = 1 + (1 if show_volume else 0) + indicator_count
        height_ratios = [3]  # Price chart
        if show_volume:
            height_ratios.append(1)
        height_ratios.extend([1] * indicator_count)

        fig, axes = create_figure(rows, self.figsize, height_ratios)
        return fig, axes, rows

    def _plot_candlesticks(self, ax, df: pd.DataFrame, symbol: str, timeframe: str, ax_index: int, total_rows: int):
        """Plot candlestick chart using common utility."""
        candle_width = calculate_candle_width(df)
        plot_candlesticks(ax, df, candle_width)
        setup_chart_style(ax, title=f"{symbol} - {timeframe}")

        if ax_index < total_rows - 1:
            ax.tick_params(labelbottom=False)
        else:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    def _plot_price_indicators(self, ax, df: pd.DataFrame, indicators: Dict):
        """Plot price indicators using common utility."""
        if "MA" in indicators:
            plot_moving_averages(ax, df, indicators["MA"])
        if "BB" in indicators:
            plot_bollinger_bands(ax, df, indicators["BB"])
        if "MA" in indicators or "BB" in indicators:
            ax.legend(loc="upper left", fontsize=8)

    def _plot_volume(self, ax, df: pd.DataFrame, ax_index: int, total_rows: int):
        """Plot volume using common utility."""
        plot_volume(ax, df)
        if ax_index < total_rows - 1:
            ax.tick_params(labelbottom=False)
        else:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    def _plot_separate_indicators(self, axes, df: pd.DataFrame, indicators: Dict, start_idx: int, total_rows: int):
        """Plot separate indicators like RSI and MACD."""
        ax_idx = 0

        # RSI
        if "RSI" in indicators and ax_idx < len(axes):
            ax = axes[ax_idx]
            rsi_config = indicators["RSI"]
            period = rsi_config.get("period", 14)
            col_name = f"RSI_{period}"

            if col_name in df.columns:
                ax.plot(df.index, df[col_name], label=f"RSI {period}", color="purple", linewidth=1.5)
                ax.axhline(y=rsi_config.get("overbought", 70), color="red", linestyle="--", alpha=0.5)
                ax.axhline(y=rsi_config.get("oversold", 30), color="green", linestyle="--", alpha=0.5)
                ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)
                setup_chart_style(ax, ylabel="RSI")
                ax.set_ylim(0, 100)
                ax.legend(loc="upper left", fontsize=8)

                if (start_idx + ax_idx) < total_rows - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
                ax_idx += 1

        # MACD
        if "MACD" in indicators and ax_idx < len(axes):
            ax = axes[ax_idx]
            if "MACD" in df.columns and "MACD_signal" in df.columns:
                ax.plot(df.index, df["MACD"], label="MACD", color="blue", linewidth=1.5)
                ax.plot(df.index, df["MACD_signal"], label="Signal", color="red", linewidth=1.5)
                if "MACD_hist" in df.columns:
                    ax.bar(df.index, df["MACD_hist"], label="Histogram", alpha=0.6, color="gray")
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
                setup_chart_style(ax, ylabel="MACD")
                ax.legend(loc="upper left", fontsize=8)

                if (start_idx + ax_idx) < total_rows - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
                ax_idx += 1

        # Hide unused axes
        for i in range(ax_idx, len(axes)):
            axes[i].axis("off")

    def _add_indicators(self, df: pd.DataFrame, indicators: Optional[Dict]) -> pd.DataFrame:
        """Calculate and add indicators to DataFrame."""
        df = df.copy()
        if indicators is None:
            indicators = {}
        for ind_type, config in indicators.items():
            try:
                if ind_type == "MA":
                    for period in config.get("periods", [20, 50, 200]):
                        df[f"MA_{period}"] = calculate_ma_series(
                            df["close"], period=period, ma_type=config.get("type", "SMA")
                        )
                elif ind_type == "RSI":
                    df[f"RSI_{config.get('period', 14)}"] = calculate_rsi_series(
                        df["close"], period=config.get("period", 14)
                    )
                elif ind_type == "MACD":
                    macd_df = calculate_macd_series(
                        df["close"],
                        fast=config.get("fast", 12),
                        slow=config.get("slow", 26),
                        signal=config.get("signal", 9),
                    )
                    df["MACD"], df["MACD_signal"], df["MACD_hist"] = (
                        macd_df["MACD"],
                        macd_df["MACD_signal"],
                        macd_df["MACD_hist"],
                    )
                elif ind_type == "BB":
                    bb_df = calculate_bollinger_bands_series(
                        df["close"], period=config.get("period", 20), std=config.get("std", 2)
                    )
                    period = config.get("period", 20)
                    df[f"BB_upper_{period}"], df[f"BB_middle_{period}"], df[f"BB_lower_{period}"] = (
                        bb_df["BB_upper"],
                        bb_df["BB_middle"],
                        bb_df["BB_lower"],
                    )
            except Exception as e:
                log_warn(f"Failed to calculate {ind_type}: {e}")
        return df
