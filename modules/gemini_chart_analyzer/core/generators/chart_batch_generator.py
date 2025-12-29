"""
Batch Chart Generator for creating composite images of multiple charts.

Groups 100 simple charts into a single batch image (10x10 grid).
"""

import pandas as pd
import math
import matplotlib
# Use non-interactive backend to avoid GUI overhead and memory leaks
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import gc

# Note: SimpleChartGenerator is not currently used, but kept for potential future use
# from modules.gemini_chart_analyzer.core.generators.simple_chart_generator import SimpleChartGenerator
from modules.common.ui.logging import log_error, log_success, log_warn
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_charts_dir


class ChartBatchGenerator:
    """Generate batch images containing multiple simple charts."""
    
    def __init__(
        self,
        charts_per_batch: int = 100,
        grid_rows: Optional[int] = None,
        grid_cols: Optional[int] = None,
        chart_size: Tuple[float, float] = (2.0, 1.5),  # inches
        dpi: int = 100
    ):
        """
        Initialize ChartBatchGenerator.
        
        Args:
            charts_per_batch: Number of charts per batch (default: 100)
            grid_rows: Number of rows in grid (auto-calculated if None)
            grid_cols: Number of columns in grid (auto-calculated if None)
            chart_size: Size of each individual chart in inches (width, height)
            dpi: DPI for output image
        """
        # Validate charts_per_batch type and value
        if not isinstance(charts_per_batch, int) or charts_per_batch <= 0:
            raise ValueError(f"charts_per_batch must be a positive integer, got {charts_per_batch}")
        
        self.charts_per_batch = charts_per_batch
        
        # Handle explicit cases for grid_rows/grid_cols None logic
        if grid_rows is None and grid_cols is None:
            grid_rows, grid_cols = self._calculate_grid_dimensions(charts_per_batch)
        elif grid_rows is None:
            if grid_cols <= 0:
                raise ValueError("grid_cols must be a positive integer.")
            if charts_per_batch % grid_cols != 0:
                raise ValueError(f"charts_per_batch ({charts_per_batch}) is not divisible by grid_cols ({grid_cols})")
            grid_rows = charts_per_batch // grid_cols
        elif grid_cols is None:
            if grid_rows <= 0:
                raise ValueError("grid_rows must be a positive integer.")
            if charts_per_batch % grid_rows != 0:
                raise ValueError(f"charts_per_batch ({charts_per_batch}) is not divisible by grid_rows ({grid_rows})")
            grid_cols = charts_per_batch // grid_rows
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.chart_size = chart_size
        self.dpi = dpi
        
        # Validate grid
        if grid_rows * grid_cols != charts_per_batch:
            raise ValueError(
                f"grid_rows ({grid_rows}) * grid_cols ({grid_cols}) must equal "
                f"charts_per_batch ({charts_per_batch})"
            )
        
        # Validate chart_size
        if not isinstance(chart_size, (tuple, list)) or len(chart_size) != 2:
            raise ValueError(
                f"chart_size must be a tuple or list of 2 elements (width, height), "
                f"got {type(chart_size).__name__}"
            )
        width, height = chart_size[0], chart_size[1]
        self._validate_positive_numeric(width, "chart_size width")
        self._validate_positive_numeric(height, "chart_size height")
        
        # Validate dpi
        self._validate_positive_numeric(dpi, "dpi")
        
        # Note: SimpleChartGenerator is not used in this class, but kept for potential future use
        # ChartBatchGenerator plots directly on axes instead
        # self.simple_generator = SimpleChartGenerator(
        #     figsize=chart_size,
        #     dpi=dpi
        # )
    
    @staticmethod
    def _calculate_grid_dimensions(charts_per_batch: int) -> Tuple[int, int]:
        """
        Calculate optimal grid_rows and grid_cols for given charts_per_batch.
        
        Tries to create a square-ish grid (rows ≈ cols) that exactly equals charts_per_batch.
        
        Args:
            charts_per_batch: Number of charts to fit in grid
            
        Returns:
            Tuple of (grid_rows, grid_cols) where grid_rows * grid_cols == charts_per_batch
        """
        
        # Calculate approximate square root
        sqrt_val = math.sqrt(charts_per_batch)
        
        # Find factors that multiply to exactly charts_per_batch
        # Optimize by iterating only up to sqrt(charts_per_batch) - O(√n) instead of O(n)
        best_rows = None
        best_cols = None
        min_diff = float('inf')
        
        # Iterate only up to sqrt(charts_per_batch) to find all factor pairs
        sqrt_n = int(math.sqrt(charts_per_batch))
        for k in range(1, sqrt_n + 1):
            if charts_per_batch % k == 0:  # k is a factor
                # Consider both factor pairs: (k, charts_per_batch//k) and (charts_per_batch//k, k)
                # Pair 1: rows=k, cols=charts_per_batch//k
                rows1, cols1 = k, charts_per_batch // k
                diff1 = abs(rows1 - cols1)
                if diff1 < min_diff:
                    min_diff = diff1
                    best_rows = rows1
                    best_cols = cols1
                
                # Pair 2: rows=charts_per_batch//k, cols=k (swapped)
                rows2, cols2 = charts_per_batch // k, k
                diff2 = abs(rows2 - cols2)
                if diff2 < min_diff:
                    min_diff = diff2
                    best_rows = rows2
                    best_cols = cols2
        
        # If no factors found (shouldn't happen for positive integers), use ceil approach
        # The unreachable fallback code is removed since best_rows will never be None for positive integers.
        
        # Defensive check: ensure grid dimensions were computed successfully
        if best_rows is None or best_cols is None:
            raise RuntimeError(
                f"Failed to compute grid dimensions for charts_per_batch={charts_per_batch}. "
                f"This should not happen for positive integers. Please check the input value."
            )
        
        return best_rows, best_cols

    def _validate_positive_numeric(
        self,
        value,
        param_name: str,
        allow_types: tuple = (int, float)
    ) -> None:
        """
        Validate that a value is numeric and positive.
        
        Args:
            value: Value to validate
            param_name: Name of parameter for error messages
            allow_types: Tuple of allowed types (default: (int, float))
            
        Raises:
            ValueError: If value is not of allowed type or is not positive
        """
        if not isinstance(value, allow_types):
            raise ValueError(
                f"{param_name} must be numeric, got {type(value).__name__}"
            )
        if value <= 0:
            raise ValueError(
                f"{param_name} must be positive, got {value}"
            )
    
    def create_batch_chart(
        self,
        symbols_data: List[Dict[str, pd.DataFrame]],
        timeframe: str,
        output_path: Optional[str] = None,
        batch_id: Optional[int] = None
    ) -> Tuple[str, bool]:
        """
        Create a batch chart image with multiple symbols.
        
        Args:
            symbols_data: List of dicts with 'symbol' and 'df' keys
                         Example: [{'symbol': 'BTC/USDT', 'df': DataFrame}, ...]
            timeframe: Timeframe string (for labeling)
            output_path: Optional output path (auto-generated if None)
            batch_id: Optional batch ID for filename
            
        Returns:
            Tuple of (output_path: str, truncated: bool):
            - output_path: Path to saved batch image
            - truncated: True if input list length exceeded charts_per_batch and was truncated,
                        False otherwise
            
        Note:
            When the input symbols_data list length exceeds charts_per_batch, the list is
            silently truncated to the first charts_per_batch items. An error is logged
            and the truncated flag in the return value will be True to indicate this occurred.
        """
        truncated = False
        if len(symbols_data) > self.charts_per_batch:
            log_warn(f"Too many symbols ({len(symbols_data)}), max is {self.charts_per_batch}")
            symbols_data = symbols_data[:self.charts_per_batch]
            truncated = True
        
        # Validate: ensure we have at least one symbol to plot
        if not symbols_data:
            raise ValueError("symbols_data cannot be empty")
        
        # Calculate total figure size
        total_width = self.chart_size[0] * self.grid_cols
        total_height = self.chart_size[1] * self.grid_rows
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_suffix = f"_batch{batch_id}" if batch_id is not None else ""
            charts_dir = get_charts_dir()
            output_dir = charts_dir / "batch"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"batch_chart_{timeframe}_{timestamp}{batch_suffix}.png"
            output_path = str(output_path)
        
        # Create main figure with style context to avoid mutating global state
        # Use context manager to ensure style is only applied to this figure
        with plt.style.context(['dark_background']):
            fig = plt.figure(figsize=(total_width, total_height), dpi=self.dpi)
            fig.patch.set_facecolor('black')
            
            # Create subplots in grid
            for idx, symbol_data in enumerate(symbols_data):
                symbol = symbol_data['symbol']
                df = symbol_data['df']
                
                # Create subplot
                ax = fig.add_subplot(self.grid_rows, self.grid_cols, idx + 1)
                
                try:
                    # Plot simple chart on this subplot
                    self._plot_simple_chart_on_axes(ax, df, symbol, timeframe)
                except Exception as e:
                    log_error(f"Error plotting {symbol}: {e}")
                    # Draw empty chart with error message
                    ax.set_facecolor('black')
                    ax.text(0.5, 0.5, f"Error\n{symbol}", 
                           transform=ax.transAxes, ha='center', va='center',
                           color='red', fontsize=8)
            
            # Fill remaining empty slots if needed
            for idx in range(len(symbols_data), self.charts_per_batch):
                ax = fig.add_subplot(self.grid_rows, self.grid_cols, idx + 1)
                ax.set_facecolor('black')
                ax.axis('off')
            
            # Tight layout
            plt.tight_layout(pad=0.5)
            
            # Save figure with error handling
            try:
                fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='black', pad_inches=0.1)
            except OSError as e:
                plt.close(fig)  # Ensure figure is closed even on error
                raise IOError(f"Failed to save batch chart to {output_path}: {e}") from e
            except Exception as e:
                plt.close(fig)  # Ensure figure is closed even on error
                raise RuntimeError(f"Unexpected error saving batch chart: {e}") from e
        
        # Close figure outside style context to ensure cleanup
        plt.close(fig)
        
        # Force garbage collection to free memory immediately
        # This helps prevent memory leaks in batch processing
        gc.collect()
        
        log_success(f"Created batch chart: {output_path}")
        return output_path, truncated
    
    def _format_symbol_label(self, symbol: str) -> str:
        """
        Format symbol string into a display label for chart annotations.
        
        Expected symbol formats:
        - "BASE/QUOTE" (e.g., "BTC/USDT", "ETH/BTC")
        - "BASE" (symbol without quote, e.g., "BTC")
        - Symbols may contain multiple slashes (e.g., "BTC/USDT/PERP")
        
        Label derivation rules:
        1. Split symbol on '/' to get parts
        2. Filter out any parts that are "USDT" (case-insensitive) for cleaner labels
        3. Join remaining parts with '_' separator
        4. If no slash found, return symbol as-is
        
        Examples:
        - "BTC/USDT" -> "BTC"
        - "ETH/BTC" -> "ETH_BTC"
        - "BTC/USDT/PERP" -> "BTC_PERP"
        - "BTCUSDT" -> "BTCUSDT" (no slash, returned as-is)
        - "BTC" -> "BTC" (no slash, returned as-is)
        
        Args:
            symbol: Symbol string in format "BASE/QUOTE" or "BASE"
            
        Returns:
            Formatted label string suitable for chart display
        """
        if not symbol:
            return symbol
        
        # Split on '/' to handle all cases robustly
        parts = symbol.split('/')
        
        # Filter out empty parts (handles edge cases like "BTC//USDT")
        parts = [part for part in parts if part]
        
        if len(parts) == 0:
            return symbol  # Return original if all parts were empty
        
        # Remove "USDT" parts (case-insensitive) for cleaner labels
        # This handles cases like "BTC/USDT" -> "BTC" and "BTC/USDT/PERP" -> "BTC_PERP"
        parts = [part for part in parts if part.upper() != 'USDT']
        
        # If all parts were USDT, return original symbol
        if len(parts) == 0:
            return symbol
        
        # Join remaining parts with '_' or return single part
        if len(parts) == 1:
            return parts[0]
        else:
            return '_'.join(parts)
    
    def _plot_simple_chart_on_axes(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ):
        """Plot a simple candlestick chart on given axes."""
        # Validate input
        if df is None or df.empty:
            ax.set_facecolor('black')
            ax.text(0.5, 0.5, f"No data\n{symbol}", 
                   transform=ax.transAxes, ha='center', va='center',
                   color='red', fontsize=8)
            return
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            ax.set_facecolor('black')
            ax.text(0.5, 0.5, f"Missing cols\n{symbol}", 
                   transform=ax.transAxes, ha='center', va='center',
                   color='red', fontsize=8)
            return
        
        # Prepare data
        df_candle = df[['open', 'high', 'low', 'close']].copy()
        
        # Ensure DatetimeIndex
        if not isinstance(df_candle.index, pd.DatetimeIndex):
            df_candle.index = pd.to_datetime(df_candle.index)
        
        # Calculate candle width
        candle_width = 0.0  # Default initialization
        if len(df_candle) > 1:
            time_diff = (df_candle.index[1] - df_candle.index[0]).total_seconds() / 86400
            candle_width = time_diff * 0.8
        elif len(df_candle) == 1:
            # Single candle case - use default width
            candle_width = 0.01
        
        # Colors
        up_color = '#00ff00'
        down_color = '#ff0000'
        
        # Plot candles
        for timestamp, row in df_candle.iterrows():
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            is_up = close_price >= open_price
            color = up_color if is_up else down_color
            
            # Draw body
            body_low = min(open_price, close_price)
            body_high = max(open_price, close_price)
            body_height = body_high - body_low
            
            if body_height > 0:
                rect = Rectangle(
                    (mdates.date2num(timestamp) - candle_width/2, body_low),
                    candle_width, body_height,
                    facecolor=color, edgecolor=color, linewidth=0.3
                )
                ax.add_patch(rect)
            else:
                ax.plot(
                    [mdates.date2num(timestamp) - candle_width/2,
                     mdates.date2num(timestamp) + candle_width/2],
                    [open_price, close_price],
                    color=color, linewidth=1.0
                )
            
            # Draw wick
            ax.plot(
                [mdates.date2num(timestamp), mdates.date2num(timestamp)],
                [low_price, high_price],
                color=color, linewidth=0.5, alpha=0.8
            )
        
        # Configure axes
        ax.set_facecolor('black')
        ax.tick_params(colors='white', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.5)
        
        # Remove x-axis labels
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        # Add symbol label
        label = self._format_symbol_label(symbol)
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
               fontsize=7, color='white', verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7, 
                        edgecolor='white', linewidth=0.3))


