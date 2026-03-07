"""
Gann Square Chart Generator.

Draws a candlestick chart overlaid with Gann Fan zones,
swing point markers, and diagonal fan lines.
Saves to PNG for sending to Gemini.
"""

from __future__ import annotations

import gc
import os
from datetime import datetime
from typing import Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend

from modules.common.domain.symbol_codec import SymbolCodec
from modules.common.ui.logging import log_success
from modules.gemini_chart_analyzer.core.plotting_utils import (
    calculate_candle_width,
    create_figure,
    plot_candlesticks,
    prepare_dataframe_for_plotting,
    setup_chart_style,
)

from .gann_calculator import GannSquareResult
from .swing_detector import SwingPoint

# Zone colors: index 0=Zone1, 1=Zone2, 2=Zone3, 3=Zone4
_DOWN_ZONE_COLORS = [
    "#333333",  # Zone 1 – darker gray (SKIP, steepest)
    "#555555",  # Zone 2 – dark gray (SKIP)
    "#ff7777",  # Zone 3 – light red (SHORT)
    "#ff2d2d",  # Zone 4 – deep red (SHORT, shallowest)
]

_UP_ZONE_COLORS = [
    "#333333",  # Zone 1 – darker gray (SKIP, steepest)
    "#555555",  # Zone 2 – dark gray (SKIP)
    "#33cc66",  # Zone 3 – light green (LONG)
    "#00a63a",  # Zone 4 – deep green (LONG, shallowest)
]

_ZONE_ALPHA = 0.18  # transparency for zone bands


class GannChartGenerator:
    """
    Generate candlestick charts with Gann Fan overlay.

    Usage:
        gen = GannChartGenerator()
        path = gen.create_chart(df, gann_result, "BTCUSDT", "4h")
    """

    DEFAULT_OUTPUT_DIR = "charts"

    def __init__(
        self,
        figsize: Tuple[int, int] = (18, 10),
        style: str = "dark_background",
        dpi: int = 150,
        output_dir: Optional[str] = None,
    ) -> None:
        self.figsize = figsize
        self.style = style
        self.dpi = dpi
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR

    def create_chart(
        self,
        df: pd.DataFrame,
        gann_result: GannSquareResult,
        symbol: str,
        timeframe: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Create and save a candlestick chart with Gann Fan overlay.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
            gann_result: Result from GannCalculator.
            symbol: Trading symbol (e.g., 'BTC/USDT').
            timeframe: Chart timeframe (e.g., '4h').
            output_path: Optional explicit output path. Auto-generated if None.

        Returns:
            Absolute path to the saved PNG file.
        """
        df = prepare_dataframe_for_plotting(df)

        if output_path is None:
            output_path = self._auto_output_path(symbol, timeframe)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with plt.style.context([self.style]):
            fig, axes = create_figure(1, self.figsize, [1])
            ax = axes[0]

            # 1. Candlestick base
            candle_width = calculate_candle_width(df)
            plot_candlesticks(ax, df, candle_width)

            # 2. Gann fan zones (diagonal lines)
            self._plot_gann_fan_zones(ax, gann_result, df)

            # 3. Swing point markers
            self._plot_swing_markers(ax, gann_result.swing_high, gann_result.swing_low)

            # 4. Current price line
            current_price = float(df["close"].iloc[-1])
            self._plot_current_price(ax, current_price, gann_result.current_zone)

            # 5. Legend
            self._add_legend(ax, gann_result, current_price)

            # 6. Title and style
            zone_label = f"Zone {gann_result.current_zone}" if gann_result.current_zone else "Out of Range"
            title = (
                f"{symbol} {timeframe} │ Gann Fan │ "
                f"Trend: {gann_result.trend} │ "
                f"{zone_label} │ "
                f"Signal: {gann_result.signal_code}"
            )
            setup_chart_style(ax, title=title)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

            plt.tight_layout()
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight", facecolor="black")

        plt.close(fig)
        gc.collect()

        log_success(f"Gann chart saved: {output_path}")
        return output_path

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _plot_gann_fan_zones(self, ax, gann_result: GannSquareResult, df: pd.DataFrame) -> None:
        """Draw diagonal fan lines for each Gann zone."""
        colors = _DOWN_ZONE_COLORS if gann_result.trend == "DOWN" else _UP_ZONE_COLORS

        # Get timestamps for x-axis
        timestamps = df.index
        candle_indices = np.arange(len(df))

        # Plot each zone as a filled area between its own boundaries
        for i, zone in enumerate(gann_result.zones):
            color = colors[i]
            is_current = zone.zone_number == gann_result.current_zone

            upper_prices = np.array([zone.upper_price_at(int(idx)) for idx in candle_indices])
            lower_prices = np.array([zone.lower_price_at(int(idx)) for idx in candle_indices])

            # Fill between the zone's own upper and lower boundaries
            ax.fill_between(
                timestamps,
                lower_prices,
                upper_prices,
                alpha=_ZONE_ALPHA * (1.8 if is_current else 1.0),
                color=color,
                zorder=1,
            )

            # Plot the fan line (upper boundary)
            ax.plot(
                timestamps,
                upper_prices,
                color=color,
                linewidth=1.0 if not is_current else 1.5,
                linestyle="--",
                alpha=0.6,
                zorder=2,
            )

            # Zone label at the last candle
            last_idx = len(df) - 1
            label_y = (upper_prices[last_idx] + lower_prices[last_idx]) / 2
            ax.text(
                timestamps[last_idx],
                label_y,
                f" {zone.label}",
                color=color,
                fontsize=8,
                va="center",
                ha="left",
                alpha=0.85,
                zorder=5,
            )

        # Plot the outermost lower boundary line (not drawn as any upper)
        # DOWN: Zone 1 (steepest, bottom)  |  UP: Zone 4 (shallowest, bottom)
        extra_idx = 0 if gann_result.trend == "DOWN" else 3
        extra_zone = gann_result.zones[extra_idx]
        extra_prices = np.array([extra_zone.lower_price_at(int(idx)) for idx in candle_indices])
        ax.plot(
            timestamps,
            extra_prices,
            color=colors[extra_idx],
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
            zorder=2,
        )

    def _plot_swing_markers(self, ax, swing_high: SwingPoint, swing_low: SwingPoint) -> None:
        """Mark Swing High with ▼ and Swing Low with ▲."""
        # Swing High marker (▼ above the candle)
        ax.annotate(
            f"▼ SWING HIGH\n{swing_high.price:,.2f}",
            xy=(swing_high.timestamp, swing_high.price),
            xytext=(swing_high.timestamp, swing_high.price * 1.005),
            color="#FFD700",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="#FFD700"),
        )

        # Swing Low marker (▲ below the candle)
        ax.annotate(
            f"▲ SWING LOW\n{swing_low.price:,.2f}",
            xy=(swing_low.timestamp, swing_low.price),
            xytext=(swing_low.timestamp, swing_low.price * 0.995),
            color="#00FFFF",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="#00FFFF"),
        )

    def _plot_current_price(self, ax, price: float, current_zone: int) -> None:
        """Draw a horizontal line at the current price."""
        ax.axhline(
            y=price,
            color="#FFFF00",
            linewidth=1.8,
            linestyle="-",
            alpha=0.9,
            zorder=4,
            label=f"Current: {price:,.2f}",
        )

    def _add_legend(self, ax, gann_result: GannSquareResult, current_price: float) -> None:
        """Add a compact legend for zone colors and current price."""
        colors = _DOWN_ZONE_COLORS if gann_result.trend == "DOWN" else _UP_ZONE_COLORS
        patches = [mpatches.Patch(color=colors[i], alpha=0.6, label=gann_result.zones[i].label) for i in range(4)]
        patches.append(mpatches.Patch(color="#FFFF00", label=f"Current Price: {float(current_price):,.4f}"))
        ax.legend(
            handles=patches,
            loc="upper left",
            fontsize=7,
            framealpha=0.6,
            facecolor="black",
            edgecolor="gray",
        )

    def _auto_output_path(self, symbol: str, timeframe: str) -> str:
        """Generate automatic output path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_symbol = SymbolCodec.sanitize_for_filename(symbol)
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, f"gann_{safe_symbol}_{timeframe}_{timestamp}.png")
