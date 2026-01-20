"""
Simple Chart Generator for batch scanning.

Render simple candlestick charts without indicators for batch processing.
"""

from typing import Tuple, Any

import matplotlib.pyplot as plt
import pandas as pd

from modules.gemini_chart_analyzer.core.plotting_utils import (
    calculate_candle_width,
    plot_candlesticks,
    prepare_dataframe_for_plotting,
)


class SimpleChartGenerator:
    """Render simple candlestick charts without indicators."""

    def __init__(
        self,
        figsize: Tuple[float, float] = (2, 1.5),  # Small size for batch images
        style: str = "dark_background",
        dpi: int = 100,  # Lower DPI for smaller file size
    ):
        """Initialize SimpleChartGenerator."""
        self.figsize = figsize
        self.style = style
        self.dpi = dpi

    def create_simple_chart(self, df: pd.DataFrame, symbol: str, timeframe: str, show_symbol_label: bool = True) -> Any:
        """Create a simple candlestick chart without indicators."""
        # Prepare data using common utility
        df_candle = prepare_dataframe_for_plotting(df)

        # Calculate candle width
        candle_width = calculate_candle_width(df_candle)

        # Use style context manager
        with plt.style.context([self.style]):
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

            # Plot candlesticks using common utility (fast vectorized version)
            plot_candlesticks(ax, df_candle, candle_width)

            # Minimal styling for batch
            ax.set_facecolor("black")
            ax.tick_params(colors="white", labelsize=6)
            for spine in ax.spines.values():
                spine.set_color("white")

            # Remove x-axis labels for cleaner batch image
            ax.set_xticks([])

            # Add symbol label if requested
            if show_symbol_label:
                label = symbol.replace("/USDT", "").replace("/", "_")
                ax.text(
                    0.02,
                    0.98,
                    label,
                    transform=ax.transAxes,
                    fontsize=8,
                    color="white",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="white", linewidth=0.5),
                )
            plt.tight_layout(pad=0.1)

        return fig
