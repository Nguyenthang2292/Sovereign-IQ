"""
Simple Chart Generator for batch scanning.

Render simple candlestick charts without indicators for batch processing.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Tuple

from modules.common.ui.logging import log_warn


class SimpleChartGenerator:
    """Render simple candlestick charts without indicators."""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (2, 1.5),  # Small size for batch images
        style: str = 'dark_background',
        dpi: int = 100  # Lower DPI for smaller file size
    ):
        """
        Initialize SimpleChartGenerator.
        
        Args:
            figsize: Figure size (width, height) in inches - small for batch
            style: Matplotlib style (default: 'dark_background')
            dpi: DPI for output (default: 100, lower for batch)
        """
        self.figsize = figsize
        self.style = style
        self.dpi = dpi
    
    def create_simple_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        show_symbol_label: bool = True
    ) -> plt.Figure:
        """
        Create a simple candlestick chart without indicators.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name (for labeling)
            timeframe: Timeframe (for labeling)
            show_symbol_label: Whether to show symbol label on chart
            
        Returns:
            matplotlib Figure object. The caller owns the returned Figure and is
            responsible for closing it when finished to prevent memory leaks.
            Close it using plt.close(fig) or fig.close(), or use an appropriate
            context manager to ensure resources are released. This is especially
            important during batch processing.
        """

        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")

        # Prepare data
        df_candle = df[['open', 'high', 'low', 'close']].copy()
        
        # Ensure DatetimeIndex
        if not isinstance(df_candle.index, pd.DatetimeIndex):
            df_candle.index = pd.to_datetime(df_candle.index)
        
        # Calculate candle width based on timeframe
        if len(df_candle) > 1:
            time_diff = (df_candle.index[1] - df_candle.index[0]).total_seconds() / 86400  # days
            candle_width = time_diff * 0.8
        else:
            candle_width = 0.01
        
        # Use style context manager to avoid mutating global matplotlib state
        with plt.style.context([self.style]):
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Colors
            up_color = '#00ff00'  # Green for up candles
            down_color = '#ff0000'  # Red for down candles
            
            # Validate and filter NaN values before conversion
            # Create mask to exclude rows with NaN in any OHLC column or timestamp
            nan_mask = (
                df_candle[['open', 'high', 'low', 'close']].notna().all(axis=1) &
                df_candle.index.notna()
            )
            
            # Count rows to be skipped
            skipped_count = (~nan_mask).sum()
            if skipped_count > 0:
                log_warn(
                    f"Warning: Skipping {skipped_count} row(s) with NaN values "
                    f"in OHLC data for {symbol} ({timeframe})"
                )
            
            # Filter DataFrame to exclude NaN rows
            df_candle_clean = df_candle[nan_mask]
            
            # Check if we have any data left after filtering
            if len(df_candle_clean) == 0:
                raise ValueError(
                    f"No valid data remaining after filtering NaN values for {symbol} ({timeframe})"
                )
            
            # Pre-convert timestamps once for performance (only valid rows)
            timestamps_num = mdates.date2num(df_candle_clean.index)
            
            # Extract OHLC as numpy arrays for faster iteration (only valid rows)
            opens = df_candle_clean['open'].values
            highs = df_candle_clean['high'].values
            lows = df_candle_clean['low'].values
            closes = df_candle_clean['close'].values
            
            # Plot candles - iterate using zip for better performance
            for timestamp_num, open_price, high_price, low_price, close_price in zip(
                timestamps_num, opens, highs, lows, closes
            ):
                # Determine color
                is_up = close_price >= open_price
                color = up_color if is_up else down_color
                
                # Draw body
                body_low = min(open_price, close_price)
                body_high = max(open_price, close_price)
                body_height = body_high - body_low
                
                if body_height > 0:
                    rect = Rectangle(
                        (timestamp_num - candle_width/2, body_low),
                        candle_width, body_height,
                        facecolor=color, edgecolor=color, linewidth=0.3
                    )
                    ax.add_patch(rect)
                else:
                    # Doji candle
                    ax.plot(
                        [timestamp_num - candle_width/2,
                         timestamp_num + candle_width/2],
                        [open_price, close_price],
                        color=color, linewidth=1.0
                    )
                
                # Draw wick
                ax.plot(
                    [timestamp_num, timestamp_num],
                    [low_price, high_price],
                    color=color, linewidth=0.5, alpha=0.8
                )
            
            # Configure axes - minimal styling for batch
            ax.set_facecolor('black')
            ax.tick_params(colors='white', labelsize=6)
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            
            # Remove x-axis labels for cleaner batch image
            ax.set_xticks([])
            ax.set_xticklabels([])
            
            # Add symbol label if requested
            if show_symbol_label:
                # Short symbol name (remove /USDT if present for cleaner display)
                label = symbol.replace('/USDT', '').replace('/', '_')
                ax.text(0.02, 0.98, label, transform=ax.transAxes,
                       fontsize=8, color='white', verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='white', linewidth=0.5))
            
            # Tight layout
            plt.tight_layout(pad=0.1)
        
        return fig

