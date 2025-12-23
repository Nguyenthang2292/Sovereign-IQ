"""
Chart Generator for creating technical analysis charts with indicators.

Tạo biểu đồ nến với các indicators như MA, RSI, Volume, etc.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from pathlib import Path

from modules.common.ui.logging import log_success


def _get_charts_dir() -> str:
    """Get the charts directory path relative to module root."""
    module_root = Path(__file__).parent.parent
    charts_dir = module_root / "charts"
    return str(charts_dir)

from modules.common.indicators import (
    calculate_ma_series,
    calculate_rsi_series,
    calculate_macd_series,
    calculate_bollinger_bands_series,
)

class ChartGenerator:
    """Tạo biểu đồ kỹ thuật với các indicators."""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (16, 10),
        style: str = 'dark_background',
        dpi: int = 150
    ):
        """
        Khởi tạo ChartGenerator.
        
        Args:
            figsize: Kích thước biểu đồ (width, height)
            style: Style của matplotlib (default: 'dark_background')
            dpi: Độ phân giải ảnh (default: 150)
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
        show_volume: bool = True,
        show_grid: bool = True
    ) -> str:
        """
        Tạo biểu đồ nến với indicators và lưu vào file.
        
        Args:
            df: DataFrame chứa OHLCV data với index là DatetimeIndex
            symbol: Tên symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            indicators: Dict các indicators cần vẽ
                Format: {
                    'MA': {'periods': [20, 50, 200], 'type': 'SMA' (hoặc 'EMA'), 'colors': ['blue', 'orange', 'red']},
                    'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
                    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
                    'BB': {'period': 20, 'std': 2}
                }
            output_path: Đường dẫn lưu file (nếu None, tự động tạo tên)
            show_volume: Hiển thị volume chart
            show_grid: Hiển thị grid
            
        Returns:
            Đường dẫn file ảnh đã lưu
        """
        if df.empty:
            raise ValueError("DataFrame rỗng, không thể tạo biểu đồ")
        
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Thiếu các cột: {missing_cols}")
        
        # Chuẩn bị dữ liệu
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                raise ValueError("DataFrame phải có DatetimeIndex hoặc cột 'timestamp'")
        
        # Tính toán indicators
        indicators = indicators or {}
        df = self._add_indicators(df, indicators)
        
        # Tạo output path nếu chưa có
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_symbol = symbol.replace('/', '_').replace(':', '_')
            charts_dir = _get_charts_dir()
            os.makedirs(charts_dir, exist_ok=True)
            output_path = os.path.join(charts_dir, f"{safe_symbol}_{timeframe}_{timestamp}.png")
        
        # Tạo thư mục nếu chưa có
        # Xử lý trường hợp output_path là bare filename (không có thư mục)
        dir_path = os.path.dirname(output_path) or '.'
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Vẽ biểu đồ
        plt.style.use(self.style)
        fig, axes, total_rows = self._create_subplots(indicators, show_volume)
        
        # Vẽ nến
        self._plot_candlesticks(axes[0], df, symbol, timeframe, ax_index=0, total_rows=total_rows)
        
        # Vẽ indicators trên price chart
        self._plot_price_indicators(axes[0], df, indicators)
        
        # Vẽ volume nếu có
        volume_ax_idx = 1
        if show_volume and 'volume' in df.columns:
            self._plot_volume(axes[volume_ax_idx], df, ax_index=volume_ax_idx, total_rows=total_rows)
            indicator_start_idx = 2
        else:
            indicator_start_idx = 1
        
        # Vẽ indicators riêng (RSI, MACD, etc.)
        indicator_axes = axes[indicator_start_idx:]
        self._plot_separate_indicators(indicator_axes, df, indicators, indicator_start_idx=indicator_start_idx, total_rows=total_rows)
        
        # Điều chỉnh layout cho toàn bộ figure sau khi tất cả subplots đã được vẽ
        plt.tight_layout()
        
        # Lưu file
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        
        log_success(f"Đã lưu biểu đồ: {output_path}")
        return output_path
    
    def _create_subplots(self, indicators: Dict, show_volume: bool) -> Tuple:
        """
        Tạo subplots cho biểu đồ với số lượng hàng động dựa trên indicators.
        
        Args:
            indicators: Dict các indicators cần vẽ
            show_volume: Có hiển thị volume chart hay không
            
        Returns:
            Tuple (fig, axes, total_rows) với axes luôn là array/list và total_rows là tổng số hàng
        """
        
        # Đếm số lượng indicators được vẽ riêng (RSI và MACD)
        separate_indicators = ['RSI', 'MACD']
        indicator_count = sum(1 for ind in separate_indicators if ind in indicators)
        
        # Tính số lượng rows:
        # - 1 row cho price chart (luôn có)
        # - +1 nếu show_volume
        # - + số lượng indicator charts riêng
        rows = 1 + (1 if show_volume else 0) + indicator_count
        
        # Đảm bảo tối thiểu 1 row
        rows = max(rows, 1)
        
        # Xây dựng height_ratios:
        # - Price chart: tỷ lệ lớn (3)
        # - Volume (nếu có): tỷ lệ nhỏ (1)
        # - Mỗi indicator chart: tỷ lệ bằng nhau (1)
        height_ratios = [3]  # Price chart
        if show_volume:
            height_ratios.append(1)  # Volume chart
        height_ratios.extend([1] * indicator_count)  # Indicator charts
        
        # Tạo subplots
        fig, axes = plt.subplots(rows, 1, figsize=self.figsize, 
                                 gridspec_kw={'height_ratios': height_ratios})
        
        # Đảm bảo axes luôn là array/list để code phía sau có thể index và hide axes
        if rows == 1:
            axes = [axes]  # Wrap single axis vào list
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else list(axes)
        
        return fig, axes, rows
    
    def _plot_candlesticks(self, ax, df: pd.DataFrame, symbol: str, timeframe: str, ax_index: int, total_rows: int):
        """Vẽ biểu đồ nến sử dụng matplotlib trực tiếp (thay thế mplfinance)."""
        # Chuẩn bị dữ liệu
        df_candle = df[['open', 'high', 'low', 'close']].copy()
        
        # Đảm bảo có DatetimeIndex
        if not isinstance(df_candle.index, pd.DatetimeIndex):
            df_candle.index = pd.to_datetime(df_candle.index)
        
        # Tính toán width của nến dựa trên khoảng thời gian
        # Sử dụng median của các time differences để xử lý dữ liệu không đều hoặc có gap
        if len(df_candle) > 1:
            # Tính toán per-row time differences
            time_diffs = df_candle.index.to_series().diff().dt.total_seconds() / 86400  # days
            # Lọc các positive diffs (loại bỏ NaN và negative values)
            positive_diffs = time_diffs[time_diffs > 0]
            
            if len(positive_diffs) > 0:
                # Sử dụng median để robust với outliers và gaps
                median_diff = positive_diffs.median()
                candle_width = median_diff * 0.8  # 80% của median time difference
            else:
                # Fallback nếu không có valid diffs
                candle_width = 0.01
        else:
            candle_width = 0.01
        
        # Màu sắc cho nến
        up_color = '#00ff00'  # Xanh cho nến tăng
        down_color = '#ff0000'  # Đỏ cho nến giảm
        
        # Vẽ từng nến
        for timestamp, row in df_candle.iterrows():
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Xác định màu: xanh nếu close >= open, đỏ nếu ngược lại
            is_up = close_price >= open_price
            color = up_color if is_up else down_color
            
            # Vẽ thân nến (body) - hình chữ nhật
            body_low = min(open_price, close_price)
            body_high = max(open_price, close_price)
            body_height = body_high - body_low
            
            # Vẽ body (rectangle) sử dụng datetime index
            if body_height > 0:
                rect = Rectangle(
                    (mdates.date2num(timestamp) - candle_width/2, body_low), 
                    candle_width, body_height,
                    facecolor=color, edgecolor=color, linewidth=0.5
                )
                ax.add_patch(rect)
            else:
                # Nến doji (open = close) - vẽ đường ngang
                ax.plot(
                    [mdates.date2num(timestamp) - candle_width/2, 
                     mdates.date2num(timestamp) + candle_width/2], 
                    [open_price, close_price], 
                    color=color, linewidth=1.5
                )
            
            # Vẽ bấc nến (wick) - đường thẳng từ high đến low
            ax.plot(
                [mdates.date2num(timestamp), mdates.date2num(timestamp)], 
                [low_price, high_price], 
                color=color, linewidth=0.8, alpha=0.8
            )
        
        # Cấu hình axes
        ax.set_title(f'{symbol} - {timeframe}', fontsize=14, fontweight='bold', color='white')
        ax.set_ylabel('Price', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, linewidth=0.5, color='gray')
        ax.set_facecolor('black')
        
        # Cấu hình x-axis với datetime - chỉ hiển thị labels trên bottom-most axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        if ax_index == total_rows - 1:
            # Bottom-most axis: hiển thị labels với rotation
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            # Các axes khác: ẩn labels
            ax.tick_params(labelbottom=False)
    
    def _plot_price_indicators(self, ax, df: pd.DataFrame, indicators: Dict):
        """Vẽ các indicators trên price chart (MA, Bollinger Bands, etc.)."""
        # Moving Averages
        if 'MA' in indicators:
            ma_config = indicators['MA']
            periods = ma_config.get('periods', [20, 50, 200])
            colors = ma_config.get('colors', ['blue', 'orange', 'red'])
            
            for period, color in zip(periods, colors):
                if f'MA_{period}' in df.columns:
                    ax.plot(df.index, df[f'MA_{period}'], 
                           label=f'MA {period}', color=color, linewidth=1.5, alpha=0.8)
        
        # Bollinger Bands
        if 'BB' in indicators:
            bb_config = indicators['BB']
            period = bb_config.get('period', 20)
            std = bb_config.get('std', 2)
            
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
        """Vẽ biểu đồ volume."""
        colors = ['#00ff00' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ff0000' 
                 for i in range(len(df))]
        ax.bar(df.index, df['volume'], color=colors, alpha=0.6)
        ax.set_ylabel('Volume', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('black')
        ax.grid(True, alpha=0.3)
        # Cấu hình x-axis với datetime - chỉ hiển thị labels trên bottom-most axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        if ax_index == total_rows - 1:
            # Bottom-most axis: hiển thị labels với rotation
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            # Các axes khác: ẩn labels
            ax.tick_params(labelbottom=False)
    
    def _plot_separate_indicators(self, axes, df: pd.DataFrame, indicators: Dict, indicator_start_idx: int, total_rows: int):
        """Vẽ các indicators riêng (RSI, MACD, etc.)."""
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
                
                # Áp dụng datetime formatting cho RSI axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                
                # Chỉ hiển thị labels trên bottom-most axis
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
                
                # Áp dụng datetime formatting cho MACD axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                
                # Chỉ hiển thị labels trên bottom-most axis
                if current_ax_index == total_rows - 1:
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                else:
                    ax.tick_params(labelbottom=False)
                
                used_axes.append(ax)
                ax_idx += 1
        
        # Ẩn các axes không sử dụng
        for i in range(ax_idx, len(axes)):
            axes[i].axis('off')
    
    def _add_indicators(self, df: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
        """Tính toán các indicators và thêm vào DataFrame."""
        df = df.copy()
        
        # Handle None indicators
        if indicators is None:
            indicators = {}
        
        # Moving Averages
        if 'MA' in indicators:
            ma_config = indicators['MA']
            periods = ma_config.get('periods', [20, 50, 200])
            ma_type = ma_config.get('type', 'SMA')  # 'SMA' or 'EMA'
            for period in periods:
                ma_series = calculate_ma_series(df['close'], period=period, ma_type=ma_type)
                df[f'MA_{period}'] = ma_series
        
        # RSI
        if 'RSI' in indicators:
            rsi_config = indicators['RSI']
            period = rsi_config.get('period', 14)
            rsi_series = calculate_rsi_series(df['close'], period=period)
            df[f'RSI_{period}'] = rsi_series
        
        # MACD
        if 'MACD' in indicators:
            macd_config = indicators['MACD']
            fast = macd_config.get('fast', 12)
            slow = macd_config.get('slow', 26)
            signal = macd_config.get('signal', 9)
            
            macd_df = calculate_macd_series(df['close'], fast=fast, slow=slow, signal=signal)
            df['MACD'] = macd_df['MACD']
            df['MACD_signal'] = macd_df['MACD_signal']
            df['MACD_hist'] = macd_df['MACD_hist']
        
        # Bollinger Bands
        if 'BB' in indicators:
            bb_config = indicators['BB']
            period = bb_config.get('period', 20)
            std = bb_config.get('std', 2)
            
            bb_df = calculate_bollinger_bands_series(df['close'], period=period, std=std)
            df[f'BB_upper_{period}'] = bb_df['BB_upper']
            df[f'BB_middle_{period}'] = bb_df['BB_middle']
            df[f'BB_lower_{period}'] = bb_df['BB_lower']
        
        return df

