"""
ATC Service - Handles ATC signal computation and data formatting for the visualizer.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import sys
import os
from pathlib import Path

current_file_path = Path(__file__).resolve()
backend_dir = current_file_path.parent

project_root = backend_dir.parent.parent.parent
project_root_absolute = project_root.resolve()

sys.path.insert(0, str(project_root_absolute))

print(f"[DEBUG] Current file: {current_file_path}")
print(f"[DEBUG] Backend dir: {backend_dir}")
print(f"[DEBUG] Project root: {project_root_absolute}")
print(f"[DEBUG] Modules exists: {(project_root_absolute / 'modules').exists()}")
print(f"[DEBUG] Python path: {sys.path[:3]}")

from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.adaptive_trend.core.analyzer import analyze_symbol
from modules.adaptive_trend.utils.config import ATCConfig


class ATCService:
    """Service for ATC analysis and data formatting."""

    def __init__(self):
        self.exchange_manager = ExchangeManager()
        self.data_fetcher = DataFetcher(self.exchange_manager)

    def get_ohlcv_data(self, symbol: str, timeframe: str = "15m", limit: int = 1500) -> Optional[Dict[str, Any]]:
        """
        Fetch OHLCV data for visualization.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '15m', '1h')
            limit: Number of candles to fetch

        Returns:
            Dictionary with OHLCV data formatted for charting
        """
        try:
            df, exchange_id = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol, limit=limit, timeframe=timeframe, check_freshness=False
            )

            if df is None or df.empty:
                return None

            ohlcv_data = []

            for idx, row in df.iterrows():
                ohlcv_data.append(
                    {
                        "x": int(idx.timestamp() * 1000),
                        "y": [float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])],
                    }
                )

            return {"symbol": symbol, "exchange": exchange_id, "timeframe": timeframe, "data": ohlcv_data}

        except Exception as e:
            print(f"Error fetching OHLCV data: {e}")
            return None

    def compute_atc_signals(
        self, symbol: str, timeframe: str = "15m", config: Optional[ATCConfig] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Compute ATC signals and Moving Averages for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe string
            config: ATCConfig object with parameters

        Returns:
            Dictionary with ATC results including MAs and signals
        """
        try:
            if config is None:
                config = ATCConfig(timeframe=timeframe)

            result = analyze_symbol(symbol=symbol, data_fetcher=self.data_fetcher, config=config)

            if result is None:
                return None

            df = result["df"]
            atc_results = result["atc_results"]

            formatted_data = {
                "symbol": symbol,
                "exchange": result["exchange_label"],
                "timeframe": timeframe,
                "current_price": float(result["current_price"]),
                "ohlcv": [],
                "moving_averages": {},
                "signals": {},
            }

            ohlcv_data = []
            for idx, row in df.iterrows():
                ohlcv_data.append(
                    {
                        "x": int(idx.timestamp() * 1000),
                        "y": [float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])],
                    }
                )
            formatted_data["ohlcv"] = ohlcv_data

            signal_data = {}

            for key, series in atc_results.items():
                if "Signal" in key:
                    data = []
                    for idx, value in series.items():
                        data.append({"x": int(idx.timestamp() * 1000), "y": float(value)})
                    signal_data[key] = data

            formatted_data["signals"] = signal_data

            return formatted_data

        except Exception as e:
            print(f"Error computing ATC signals: {e}")
            import traceback

            traceback.print_exc()
            return None

    def list_available_symbols(self) -> List[str]:
        """
        List available trading symbols from Binance.

        Returns:
            List of symbol strings
        """
        try:
            symbols = self.data_fetcher.list_binance_futures_symbols(exclude_symbols=set(), max_candidates=50)
            return symbols
        except Exception as e:
            print(f"Error listing symbols: {e}")
            return []

    def get_moving_averages(
        self, symbol: str, timeframe: str = "15m", config: Optional[ATCConfig] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get all Moving Averages computed by ATC for a symbol.

        Args:
            symbol: Trading pair
            timeframe: Timeframe string
            config: ATCConfig object

        Returns:
            Dictionary with all MA series
        """
        try:
            from modules.adaptive_trend.core.compute_moving_averages import set_of_moving_averages

            if config is None:
                config = ATCConfig(timeframe=timeframe)

            df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol, limit=config.limit, timeframe=config.timeframe
            )

            if df is None or df.empty:
                return None

            close_prices = df["close"]

            ma_types = ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]
            ma_data = {}

            for ma_type in ma_types:
                length = getattr(config, f"{ma_type.lower()}_len")

                ma_tuple = set_of_moving_averages(
                    length=length, source=close_prices, ma_type=ma_type, robustness=config.robustness
                )

                if ma_tuple:
                    mas = {
                        "MA": ma_tuple[0],
                        "MA1": ma_tuple[1],
                        "MA2": ma_tuple[2],
                        "MA3": ma_tuple[3],
                        "MA4": ma_tuple[4],
                        "MA_1": ma_tuple[5],
                        "MA_2": ma_tuple[6],
                        "MA_3": ma_tuple[7],
                        "MA_4": ma_tuple[8],
                    }

                    for ma_name, series in mas.items():
                        if series is not None:
                            data = []
                            for idx, value in series.items():
                                if not pd.isna(value):
                                    data.append({"x": int(idx.timestamp() * 1000), "y": float(value)})
                            ma_data[f"{ma_type}_{ma_name}"] = data

            return ma_data

        except Exception as e:
            print(f"Error computing moving averages: {e}")
            import traceback

            traceback.print_exc()
            return None
