"""
Chart Generator for creating technical analysis charts with indicators.

Creates candlestick charts with indicators such as MA, RSI, Volume, etc.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend to avoid GUI overhead and memory leaks
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, LineCollection
from typing import Optional, Dict, Tuple
from datetime import datetime
import gc
import itertools
import traceback

from modules.common.ui.logging import log_success, log_warn
from modules.common.indicators import (
    calculate_ma_series,
    calculate_rsi_series,
    calculate_macd_series,
    calculate_bollinger_bands_series,
)
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_charts_dir

class ChartGenerator:
    """Generate technical charts with indicators."""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (16, 10),
        style: str = 'dark_background',
        dpi: int = 150
    ):
        """
        Initialize ChartGenerator.
        
        Args:
            figsize: Size of the chart (width, height)
            style: Matplotlib style (default: 'dark_background')
            dpi: Image resolution (default: 150)
        """
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
        show_volume: bool = True
    ) -> str:
        """
        Create a candlestick chart with indicators and save to a file.
        
        Args:
            df: DataFrame containing OHLCV data with a DatetimeIndex
            symbol: Symbol name (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            indicators: Dict of indicators to plot
                Format: {
                    'MA': {'periods': [20, 50, 200], 'type': 'SMA' (or 'EMA'), 'colors': ['blue', 'orange', 'red']},
                    'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
                    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
                    'BB': {'period': 20, 'std': 2}
                }
            output_path: File path to save (if None, automatically generates a name)
            show_volume: Display volume chart
            
        Returns:
            Path of the saved chart image file
        """
        if df.empty:
            raise ValueError("Empty DataFrame, cannot create chart")
        
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Prepare data
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                raise ValueError("DataFrame must have a DatetimeIndex or a 'timestamp' column")
        
        # Calculate indicators
        indicators = indicators or {}
        df = self._add_indicators(df, indicators)
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_symbol = symbol.replace('/', '_').replace(':', '_')
            charts_dir = get_charts_dir()
            os.makedirs(str(charts_dir), exist_ok=True)
            output_path = os.path.join(str(charts_dir), f"{safe_symbol}_{timeframe}_{timestamp}.png")
        
        # Make sure the directory exists (only create if path has a directory component)
        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Plot chart within style context to avoid mutating global state
        with plt.style.context([self.style]):
            fig, axes, total_rows = self._create_subplots(indicators, show_volume)
            
            # Plot candlesticks
            self._plot_candlesticks(axes[0], df, symbol, timeframe, ax_index=0, total_rows=total_rows)
            
            # Plot indicators on price chart
            self._plot_price_indicators(axes[0], df, indicators)
            
            # Plot volume if available
            volume_ax_idx = 1
            if show_volume and 'volume' in df.columns:
                self._plot_volume(axes[volume_ax_idx], df, ax_index=volume_ax_idx, total_rows=total_rows)
                indicator_start_idx = 2
            else:
                indicator_start_idx = 1
            
            # Plot separate indicators (RSI, MACD, etc.)
            indicator_axes = axes[indicator_start_idx:]
            self._plot_separate_indicators(indicator_axes, df, indicators, indicator_start_idx=indicator_start_idx, total_rows=total_rows)
            
            # Adjust layout for the entire figure after all subplots have been drawn
            plt.tight_layout()
            
            # Save file
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='black')
        
        # Close figure outside style context to ensure cleanup
        plt.close(fig)
        
        # Force garbage collection to free memory immediately
        gc.collect()
        
        log_success(f"Chart saved: {output_path}")
        return output_path
    
    def _create_subplots(self, indicators: Optional[Dict], show_volume: bool) -> Tuple:
        """
        Create subplots for the chart with dynamic number of rows based on indicators.
        
        Args:
            indicators: Dict of indicators to plot (can be None)
            show_volume: Whether to display the volume chart
            
        Returns:
            Tuple (fig, axes, total_rows) where axes is always an array/list and total_rows is row count
        """
        
        # Count indicators that are plotted on separate axes (RSI and MACD)
        separate_indicators = ['RSI', 'MACD']
        if indicators is None:
            indicators = {}
        indicator_count = sum(1 for ind in separate_indicators if ind in indicators)
        
        # Calculate number of rows:
        # - 1 row for price chart (always present)
        # - +1 if show_volume
        # - + number of indicator charts
        rows = 1 + (1 if show_volume else 0) + indicator_count
        # rows always >= 1 because base is 1 + non-negative values
        
        # Build height_ratios:
        # - Price chart: large ratio (3)
        # - Volume (if present): small ratio (1)
        # - Each indicator chart: equal ratio (1)
        height_ratios = [3]  # Price chart
        if show_volume:
            height_ratios.append(1)  # Volume chart
        height_ratios.extend([1] * indicator_count)  # Indicator charts
        
        # Create subplots
        fig, axes = plt.subplots(rows, 1, figsize=self.figsize, 
                                 gridspec_kw={'height_ratios': height_ratios})
        
        # Ensure axes is always an array/list for indexing and hiding axes
        if rows == 1:
            axes = [axes]  # Wrap single axis as list
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else list(axes)
        
        return fig, axes, rows
    
    def _plot_candlesticks(self, ax, df: pd.DataFrame, symbol: str, timeframe: str, ax_index: int, total_rows: int):
        """Plot candlestick chart using matplotlib directly (replace mplfinance)."""
        # Prepare data
        df_candle = df[['open', 'high', 'low', 'close']].copy()
        
        # Ensure DatetimeIndex
        if not isinstance(df_candle.index, pd.DatetimeIndex):
            df_candle.index = pd.to_datetime(df_candle.index)
        
        # Calculate width of candles based on time interval
        # Use median time difference to handle irregular data or gaps
        if len(df_candle) > 1:
            # Ensure index is sorted for correct time diff calculation
            if not df_candle.index.is_monotonic_increasing:
                # Log warning with context before sorting
                row_count = len(df_candle)
                index_sample_size = min(5, row_count)
                first_indices = df_candle.index[:index_sample_size].strftime('%Y-%m-%d %H:%M:%S').tolist()
                last_indices = df_candle.index[-index_sample_size:].strftime('%Y-%m-%d %H:%M:%S').tolist()
                log_warn(
                    f"DataFrame index is not monotonic for {symbol} ({timeframe}). "
                    f"Performing defensive sort. "
                    f"Row count: {row_count}. "
                    f"First {index_sample_size} indices: {first_indices}. "
                    f"Last {index_sample_size} indices: {last_indices}."
                )
                df_candle = df_candle.sort_index()
            
            time_diffs = df_candle.index.to_series().diff().dt.total_seconds() / 86400  # days
            # Filter positive diffs (remove NaN and negative values)
            positive_diffs = time_diffs[time_diffs > 0]
            
            if len(positive_diffs) > 0:
                median_diff = positive_diffs.median()
                # Check NaN to avoid error when plotting
                if pd.notna(median_diff) and median_diff > 0:
                    candle_width = median_diff * 0.8  # 80% of median time difference
                else:
                    # Fallback if median is NaN or <= 0
                    candle_width = 0.01
            else:
                # Fallback if no valid diffs
                candle_width = 0.01
        else:
            candle_width = 0.01
            
        # Candle colors
        up_color = '#00ff00'  # Green for bullish candles        
        down_color = '#ff0000'  # Red for bearish candles
        
        # Convert datetime index to numeric values
        x_numeric = mdates.date2num(df_candle.index)
        
        # Vector calculations
        opens = df_candle['open'].values
        highs = df_candle['high'].values
        lows = df_candle['low'].values
        closes = df_candle['close'].values
        
        # Determine color for each candle: green if close >= open, red otherwise
        is_up = closes >= opens
        colors = np.where(is_up, up_color, down_color)
        
        # Calculate body (candle body)
        body_lows = np.minimum(opens, closes)
        body_highs = np.maximum(opens, closes)
        body_heights = body_highs - body_lows
        
        # Separate candles with body (body_height > 0) and doji candles (body_height = 0)
        has_body = body_heights > 0
        is_doji = ~has_body
        
        # List of Rectangle for bodies (only for candles with body)
        body_indices = np.where(has_body)[0]
        rectangles = [
            Rectangle(
                (x_numeric[i] - candle_width/2, body_lows[i]),
                candle_width,
                body_heights[i]
            )
            for i in body_indices
        ]
        body_colors = colors[body_indices].tolist()
        
        # Add PatchCollection for body
        if rectangles:
            body_collection = PatchCollection(rectangles, facecolors=body_colors, 
                                             edgecolors=body_colors, linewidths=0.5)
            ax.add_collection(body_collection)
        
        # Draw doji candles (open = close) - horizontal line
        if np.any(is_doji):
            doji_indices = np.where(is_doji)[0]
            doji_x_starts = x_numeric[doji_indices] - candle_width/2
            doji_x_ends = x_numeric[doji_indices] + candle_width/2
            doji_prices = opens[doji_indices]
            doji_colors = colors[doji_indices]
            
            # Create LineCollection for doji candles (vectorized)
            n_doji = len(doji_indices)
            doji_segments = np.zeros((n_doji, 2, 2))
            doji_segments[:, 0, 0] = doji_x_starts
            doji_segments[:, 0, 1] = doji_prices
            doji_segments[:, 1, 0] = doji_x_ends
            doji_segments[:, 1, 1] = doji_prices
            doji_collection = LineCollection(doji_segments, colors=doji_colors, linewidths=1.5)
            ax.add_collection(doji_collection)
        
        # Draw candle wicks - vertical lines from high to low for all candles (vectorized)
        n_candles = len(df_candle)
        wick_segments = np.zeros((n_candles, 2, 2))
        wick_segments[:, 0, 0] = x_numeric
        wick_segments[:, 0, 1] = lows
        wick_segments[:, 1, 0] = x_numeric
        wick_segments[:, 1, 1] = highs
        wick_collection = LineCollection(wick_segments, colors=colors, linewidths=0.8, alpha=0.8)
        ax.add_collection(wick_collection)
        
        # Update axis limits after adding collections
        ax.autoscale_view()
         
        # Configure axes
        ax.set_title(f'{symbol} - {timeframe}', fontsize=14, fontweight='bold', color='white')        
        ax.set_ylabel('Price', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, linewidth=0.5, color='gray')
        ax.set_facecolor('black')
        
        # Set x-axis as datetime - only show labels on the bottom-most axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        if ax_index == total_rows - 1:
            # Bottom-most axis: show labels with rotation
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            # Other axes: hide labels
            ax.tick_params(labelbottom=False)
    
    def _plot_price_indicators(self, ax, df: pd.DataFrame, indicators: Dict):
        """Plot indicators on price chart (MA, Bollinger Bands, etc.)."""
        # Moving Averages
        if 'MA' in indicators:
            ma_config = indicators['MA']
            periods = ma_config.get('periods', [20, 50, 200])
            colors = ma_config.get('colors', ['blue', 'orange', 'red'])
            
            # Warn if color count is less than periods count
            if len(colors) < len(periods):
                log_warn(f"Color count ({len(colors)}) is less than number of periods ({len(periods)}). Will use default color or cycle colors.")
            
            # Use default color if colors is empty, else cycle through colors
            if not colors:
                default_color = 'cyan'
                color_iter = itertools.cycle([default_color])
            else:
                color_iter = itertools.cycle(colors)
            
            # Loop through each period and assign respective color
            for period in periods:
                color = next(color_iter)
                if f'MA_{period}' in df.columns:
                    ax.plot(df.index, df[f'MA_{period}'], 
                           label=f'MA {period}', color=color, linewidth=1.5, alpha=0.8)
        
        # Bollinger Bands
        if 'BB' in indicators:
            bb_config = indicators['BB']
            period = bb_config.get('period', 20)
            
            if f'BB_upper_{period}' in df.columns:
                ax.plot(df.index, df[f'BB_upper_{period}'], 
                       label=f'BB Upper', color='cyan', linewidth=1, alpha=0.6, linestyle='--')
            if f'BB_lower_{period}' in df.columns:
                ax.plot(df.index, df[f'BB_lower_{period}'], 
                       label=f'BB Lower', color='cyan', linewidth=1, alpha=0.6, linestyle='--')
            if f'BB_middle_{period}' in df.columns:
                ax.plot(df.index, df[f'BB_middle_{period}'], 
                       label=f'BB Middle', color='yellow', linewidth=1, alpha=0.4)
        
        if 'MA' in indicators or 'BB' in indicators:
            ax.legend(loc='upper left', fontsize=8)
    
    def _plot_volume(self, ax, df: pd.DataFrame, ax_index: int, total_rows: int):
        """Plot volume chart."""
        colors = np.where(df['close'] >= df['open'], '#00ff00', '#ff0000')
        ax.bar(df.index, df['volume'], color=colors, alpha=0.6)
        ax.set_ylabel('Volume', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('black')
        ax.grid(True, alpha=0.3)
        # Set x-axis as datetime - only show labels on bottom-most axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        if ax_index == total_rows - 1:
            # Bottom-most axis: show labels with rotation
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            # Other axes: hide labels
            ax.tick_params(labelbottom=False)
    
    def _plot_separate_indicators(self, axes, df: pd.DataFrame, indicators: Dict, indicator_start_idx: int, total_rows: int):
        """Plot separate indicators (RSI, MACD, etc.)."""
        ax_idx = 0
        used_axes = []  # Track which axes are used for indicators
        
        # RSI
        if 'RSI' in indicators:
            rsi_config = indicators['RSI']
            period = rsi_config.get('period', 14)
            overbought = rsi_config.get('overbought', 70)
            oversold = rsi_config.get('oversold', 30)
            
            if f'RSI_{period}' in df.columns and ax_idx < len(axes):
                ax = axes[ax_idx]
                current_ax_index = indicator_start_idx + ax_idx
                ax.plot(df.index, df[f'RSI_{period}'], label=f'RSI {period}', color='purple', linewidth=1.5)
                ax.axhline(y=overbought, color='red', linestyle='--', alpha=0.5, label='Overbought')
                ax.axhline(y=oversold, color='green', linestyle='--', alpha=0.5, label='Oversold')
                ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
                ax.set_ylabel('RSI', color='white')
                ax.set_ylim(0, 100)
                ax.set_facecolor('black')
                ax.legend(loc='upper left', fontsize=8)
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3)
                
                # Apply datetime formatting for RSI axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                
                # Only show labels on the bottom-most axis
                if current_ax_index == total_rows - 1:
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                else:
                    ax.tick_params(labelbottom=False)
                
                used_axes.append(ax)
                ax_idx += 1
        
        # MACD
        if 'MACD' in indicators and ax_idx < len(axes):
            macd_config = indicators['MACD']
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                ax = axes[ax_idx]
                current_ax_index = indicator_start_idx + ax_idx
                ax.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=1.5)
                ax.plot(df.index, df['MACD_signal'], label='Signal', color='red', linewidth=1.5)
                if 'MACD_hist' in df.columns:
                    ax.bar(df.index, df['MACD_hist'], label='Histogram', alpha=0.6, color='gray')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
                ax.set_ylabel('MACD', color='white')
                ax.set_facecolor('black')
                ax.legend(loc='upper left', fontsize=8)
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3)
                
                # Apply datetime formatting for MACD axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                
                # Only show labels on the bottom-most axis
                if current_ax_index == total_rows - 1:
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                else:
                    ax.tick_params(labelbottom=False)
                
                used_axes.append(ax)
                ax_idx += 1
        
        # Hide axes that are not used
        for i in range(ax_idx, len(axes)):
            axes[i].axis('off')
    
    def _add_indicators(self, df: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
        """Calculate indicators and add them to the DataFrame."""
        df = df.copy()
        
        # Handle None indicators
        if indicators is None:
            indicators = {}
        
        # Moving Averages
        if 'MA' in indicators:
            try:
                ma_config = indicators['MA']
                periods = ma_config.get('periods', [20, 50, 200])
                ma_type = ma_config.get('type', 'SMA')  # 'SMA' or 'EMA'
                for period in periods:
                    ma_series = calculate_ma_series(df['close'], period=period, ma_type=ma_type)
                    df[f'MA_{period}'] = ma_series
            except Exception as e:
                log_warn(f"Failed to calculate MA indicator: {str(e)}\n{traceback.format_exc()}")
        
        # RSI
        if 'RSI' in indicators:
            try:
                rsi_config = indicators['RSI']
                period = rsi_config.get('period', 14)
                rsi_series = calculate_rsi_series(df['close'], period=period)
                df[f'RSI_{period}'] = rsi_series
            except Exception as e:
                log_warn(f"Failed to calculate RSI indicator: {str(e)}\n{traceback.format_exc()}")
        
        # MACD
        if 'MACD' in indicators:
            try:
                macd_config = indicators['MACD']
                fast = macd_config.get('fast', 12)
                slow = macd_config.get('slow', 26)
                signal = macd_config.get('signal', 9)
                
                macd_df = calculate_macd_series(df['close'], fast=fast, slow=slow, signal=signal)
                df['MACD'] = macd_df['MACD']
                df['MACD_signal'] = macd_df['MACD_signal']
                df['MACD_hist'] = macd_df['MACD_hist']
            except Exception as e:
                log_warn(f"Failed to calculate MACD indicator: {str(e)}\n{traceback.format_exc()}")
        
        # Bollinger Bands
        if 'BB' in indicators:
            try:
                bb_config = indicators['BB']
                period = bb_config.get('period', 20)
                std = bb_config.get('std', 2)
                
                bb_df = calculate_bollinger_bands_series(df['close'], period=period, std=std)
                df[f'BB_upper_{period}'] = bb_df['BB_upper']
                df[f'BB_middle_{period}'] = bb_df['BB_middle']
                df[f'BB_lower_{period}'] = bb_df['BB_lower']
            except Exception as e:
                log_warn(f"Failed to calculate BB indicator: {str(e)}\n{traceback.format_exc()}")
        
        return df

