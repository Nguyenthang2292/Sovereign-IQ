"""
Multi-Timeframe Batch Chart Generator for creating composite images with multiple timeframes per symbol.

Mỗi symbol có sub-charts cho các timeframes khác nhau trong một grid layout.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import gc

from modules.gemini_chart_analyzer.core.batch_chart_generator import BatchChartGenerator
from modules.common.ui.logging import log_info, log_error, log_success


def _get_charts_dir() -> Path:
    """Get the charts directory path relative to module root."""
    module_root = Path(__file__).parent.parent
    charts_dir = module_root / "charts"
    return charts_dir


class MultiTFBatchChartGenerator(BatchChartGenerator):
    """Generate batch images with multiple timeframes per symbol."""
    
    def __init__(
        self,
        charts_per_batch: int = 25,  # Reduced because each symbol takes more space
        timeframes_per_symbol: int = 4,  # Default: 4 timeframes per symbol
        chart_size: Tuple[float, float] = (3.0, 2.0),  # Larger per chart
        dpi: int = 100
    ):
        """
        Initialize MultiTFBatchChartGenerator.
        
        Args:
            charts_per_batch: Number of symbols per batch (default: 25, because each symbol has multiple TFs)
            timeframes_per_symbol: Number of timeframes to show per symbol (default: 4)
            chart_size: Size of each individual chart in inches (width, height)
            dpi: DPI for output image
        """
        # Calculate actual number of subplots needed (symbols * timeframes per symbol)
        total_subplots = charts_per_batch * timeframes_per_symbol
        
        # Calculate grid dimensions to fit total_subplots
        # We need total_rows * total_cols == total_subplots
        # Each symbol has timeframes_per_symbol sub-charts
        # total_rows must be a multiple of tf_rows
        # total_cols must be a multiple of tf_cols
        
        # First, calculate sub-grid for timeframes (e.g., 2x2 for 4 TFs)
        tf_rows = int(timeframes_per_symbol ** 0.5)
        tf_cols = (timeframes_per_symbol + tf_rows - 1) // tf_rows
        
        # Calculate optimal grid dimensions directly from total_subplots
        # Find factors of total_subplots that work well with tf_rows and tf_cols
        # Try to make it roughly square
        optimal_total_rows = int(total_subplots ** 0.5)
        # Round to nearest multiple of tf_rows
        optimal_total_rows = ((optimal_total_rows + tf_rows - 1) // tf_rows) * tf_rows
        optimal_total_cols = (total_subplots + optimal_total_rows - 1) // optimal_total_rows
        # Round to nearest multiple of tf_cols
        optimal_total_cols = ((optimal_total_cols + tf_cols - 1) // tf_cols) * tf_cols
        
        # Verify: if not equal, try adjusting
        if optimal_total_rows * optimal_total_cols != total_subplots:
            # Try the other way: fix total_cols first
            optimal_total_cols = int(total_subplots ** 0.5)
            optimal_total_cols = ((optimal_total_cols + tf_cols - 1) // tf_cols) * tf_cols
            optimal_total_rows = (total_subplots + optimal_total_cols - 1) // optimal_total_cols
            optimal_total_rows = ((optimal_total_rows + tf_rows - 1) // tf_rows) * tf_rows
        
        # If still not equal, use exact calculation
        if optimal_total_rows * optimal_total_cols != total_subplots:
            # Find the closest valid grid
            best_diff = float('inf')
            best_rows = optimal_total_rows
            best_cols = optimal_total_cols
            
            # Try different multiples of tf_rows for total_rows
            for rows_mult in range(1, (total_subplots // tf_rows) + 2):
                test_total_rows = rows_mult * tf_rows
                if test_total_rows > total_subplots:
                    break
                test_total_cols = (total_subplots + test_total_rows - 1) // test_total_rows
                test_total_cols = ((test_total_cols + tf_cols - 1) // tf_cols) * tf_cols
                diff = abs(test_total_rows * test_total_cols - total_subplots)
                if diff < best_diff:
                    best_diff = diff
                    best_rows = test_total_rows
                    best_cols = test_total_cols
            
            optimal_total_rows = best_rows
            optimal_total_cols = best_cols
        
        total_rows = optimal_total_rows
        total_cols = optimal_total_cols
        
        # Calculate symbols grid from total grid
        symbols_per_row = total_rows // tf_rows
        symbols_per_col = total_cols // tf_cols
        
        # Initialize parent with calculated grid
        # Pass total_subplots as charts_per_batch to satisfy parent validation
        super().__init__(
            charts_per_batch=total_subplots,
            grid_rows=total_rows,
            grid_cols=total_cols,
            chart_size=chart_size,
            dpi=dpi
        )
        
        # Store original charts_per_batch (number of symbols, not subplots)
        self.symbols_per_batch = charts_per_batch
        self.timeframes_per_symbol = timeframes_per_symbol
        self.symbols_per_row = symbols_per_row
        self.symbols_per_col = symbols_per_col
        self.tf_rows = tf_rows
        self.tf_cols = tf_cols
    
    def create_multi_tf_batch_chart(
        self,
        symbols_data: Dict[str, Dict[str, pd.DataFrame]],  # {symbol: {timeframe: df}}
        timeframes: List[str],
        output_path: Optional[str] = None,
        batch_id: Optional[int] = None
    ) -> Tuple[str, bool]:
        """
        Create a batch chart with multiple timeframes per symbol.
        
        Args:
            symbols_data: Dict mapping symbol -> dict of timeframes -> DataFrames
                Example: {
                    'BTC/USDT': {
                        '15m': DataFrame,
                        '1h': DataFrame,
                        '4h': DataFrame,
                        '1d': DataFrame
                    },
                    ...
                }
            timeframes: List of timeframes to include (must match keys in symbols_data)
            output_path: Optional output path (auto-generated if None)
            batch_id: Optional batch ID for filename
        
        Returns:
            Tuple of (output_path: str, truncated: bool)
        """
        # Limit to symbols_per_batch symbols (not charts_per_batch which is now total_subplots)
        symbol_list = list(symbols_data.keys())[:self.symbols_per_batch]
        truncated = len(symbols_data) > self.symbols_per_batch
        
        if truncated:
            log_error(f"Too many symbols ({len(symbols_data)}), max is {self.symbols_per_batch}")
        
        if not symbol_list:
            raise ValueError("symbols_data cannot be empty")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_suffix = f"_batch{batch_id}" if batch_id is not None else ""
            charts_dir = _get_charts_dir()
            output_dir = charts_dir / "batch"
            output_dir.mkdir(parents=True, exist_ok=True)
            tf_str = "_".join(timeframes)
            output_path = output_dir / f"batch_chart_multi_tf_{tf_str}_{timestamp}{batch_suffix}.png"
            output_path = str(output_path)
        
        # Calculate total figure size
        total_width = self.chart_size[0] * self.grid_cols
        total_height = self.chart_size[1] * self.grid_rows
        
        # Create main figure
        with plt.style.context(['dark_background']):
            fig = plt.figure(figsize=(total_width, total_height), dpi=self.dpi)
            fig.patch.set_facecolor('black')
            
            # Plot each symbol with its timeframes
            for symbol_idx, symbol in enumerate(symbol_list):
                if symbol not in symbols_data:
                    continue
                
                symbol_tf_data = symbols_data[symbol]
                
                # Calculate symbol position in main grid
                symbol_row = symbol_idx // self.symbols_per_col
                symbol_col = symbol_idx % self.symbols_per_col
                
                # Plot timeframes for this symbol
                for tf_idx, timeframe in enumerate(timeframes[:self.timeframes_per_symbol]):
                    if timeframe not in symbol_tf_data:
                        continue
                    
                    df = symbol_tf_data[timeframe]
                    
                    # Calculate subplot position
                    # Main grid position: (symbol_row * tf_rows + tf_row, symbol_col * tf_cols + tf_col)
                    tf_row = tf_idx // self.tf_cols
                    tf_col = tf_idx % self.tf_cols
                    
                    subplot_row = symbol_row * self.tf_rows + tf_row
                    subplot_col = symbol_col * self.tf_cols + tf_col
                    
                    # Calculate 1-based subplot index
                    subplot_idx = subplot_row * self.grid_cols + subplot_col + 1
                    
                    # Create subplot
                    ax = fig.add_subplot(self.grid_rows, self.grid_cols, subplot_idx)
                    
                    try:
                        # Plot simple chart
                        self._plot_simple_chart_on_axes(ax, df, symbol, timeframe)
                    except Exception as e:
                        log_error(f"Error plotting {symbol} {timeframe}: {e}")
                        ax.set_facecolor('black')
                        ax.text(0.5, 0.5, f"Error\n{symbol}\n{timeframe}", 
                               transform=ax.transAxes, ha='center', va='center',
                               color='red', fontsize=6)
            
            # Fill remaining empty slots
            total_subplots = self.grid_rows * self.grid_cols
            for idx in range(len(symbol_list) * self.timeframes_per_symbol, total_subplots):
                ax = fig.add_subplot(self.grid_rows, self.grid_cols, idx + 1)
                ax.set_facecolor('black')
                ax.axis('off')
            
            # Tight layout
            plt.tight_layout(pad=0.3)
            
            # Save figure
            try:
                fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='black', pad_inches=0.1)
            except Exception as e:
                plt.close(fig)
                raise IOError(f"Failed to save multi-TF batch chart: {e}") from e
        
        plt.close(fig)
        gc.collect()
        
        log_success(f"Created multi-TF batch chart: {output_path}")
        return (output_path, truncated)

