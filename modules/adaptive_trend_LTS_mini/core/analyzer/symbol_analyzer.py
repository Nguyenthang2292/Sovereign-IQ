"""
Symbol Analyzer for ATC Analysis.

This module orchestrates the complete analysis workflow for a single symbol.
"""

import traceback
from typing import TYPE_CHECKING, Any, Dict, Optional

from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig
from modules.common.system import get_memory_manager
from modules.common.utils import log_error, log_progress

from .data_provider import DataProvider
from .price_source_selector import PriceSourceSelector

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher

__all__ = ["SymbolAnalyzer"]


class SymbolAnalyzer:
    """
    Orchestrates the complete ATC analysis workflow for a symbol.

    Responsible for:
    - Coordinating data fetching, price selection, and signal computation
    - Managing memory operations
    - Formatting analysis results
    - Error handling and logging
    """

    def __init__(self, data_fetcher: "DataFetcher"):
        """
        Initialize SymbolAnalyzer.

        Args:
            data_fetcher: DataFetcher instance for market data
        """
        self.data_provider = DataProvider(data_fetcher)
        self.price_selector = PriceSourceSelector()
        self.mem_manager = get_memory_manager()

    def analyze(
        self,
        symbol: str,
        config: ATCConfig,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a single symbol using ATC.

        This method orchestrates the complete analysis workflow:
        1. Fetch OHLCV data
        2. Select and validate price source
        3. Compute ATC signals
        4. Format and return results

        Args:
            symbol: Symbol to analyze
            config: ATCConfig containing all ATC parameters

        Returns:
            Dictionary containing analysis results with keys:
                - symbol: Symbol name
                - df: OHLCV DataFrame
                - atc_results: ATC signals dictionary
                - current_price: Current price
                - exchange_label: Exchange identifier
            Returns None if analysis failed.
        """
        with self.mem_manager.safe_memory_operation(f"analyze_symbol:{symbol}"):
            try:
                # Step 1: Fetch data
                data_result = self.data_provider.fetch_symbol_data(
                    symbol=symbol,
                    timeframe=config.timeframe,
                    limit=config.limit,
                )

                if data_result is None:
                    return None

                df, exchange_label = data_result

                # Step 2: Select price source
                price_result = self.price_selector.select_price_source(
                    df=df,
                    calculation_source=config.calculation_source,
                    symbol=symbol,
                )

                if price_result is None:
                    return None

                price_series, current_price = price_result

                # Step 3: Compute ATC signals
                log_progress(
                    f"Calculating ATC signals for {symbol} "
                    f"using {config.calculation_source} prices..."
                )

                atc_results = self._compute_signals(price_series, config)

                # Step 4: Format and return results
                return {
                    "symbol": symbol,
                    "df": df,
                    "atc_results": atc_results,
                    "current_price": current_price,
                    "exchange_label": exchange_label,
                }

            except Exception as e:
                log_error(f"Error analyzing {symbol}: {type(e).__name__}: {e}")
                log_error(f"Traceback: {traceback.format_exc()}")
                return None

    def _compute_signals(self, price_series, config: ATCConfig) -> Dict[str, Any]:
        """
        Compute ATC signals from price series.

        Args:
            price_series: pandas Series containing price data
            config: ATCConfig with signal computation parameters

        Returns:
            Dictionary containing ATC signal results
        """
        return compute_atc_signals(
            prices=price_series,
            src=None,  # Use selected price source
            ema_len=config.ema_len,
            hma_len=config.hma_len,
            wma_len=config.wma_len,
            dema_len=config.dema_len,
            lsma_len=config.lsma_len,
            kama_len=config.kama_len,
            ema_w=config.ema_w,
            hma_w=config.hma_w,
            wma_w=config.wma_w,
            dema_w=config.dema_w,
            lsma_w=config.lsma_w,
            kama_w=config.kama_w,
            robustness=config.robustness,
            lambda_param=config.lambda_param,
            decay_rate=config.decay,
            cutout=config.cutout,
            long_threshold=config.long_threshold,
            short_threshold=config.short_threshold,
            parallel_l1=config.parallel_l1,
            parallel_l2=config.parallel_l2,
            use_rust_backend=config.use_rust_backend,
            equity_floor=config.equity_floor,
        )
