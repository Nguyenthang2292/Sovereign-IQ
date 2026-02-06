"""
Manual Mode Execution for ATC Analysis.

This module handles single symbol analysis and result display.
"""

from argparse import Namespace
from typing import Optional, TYPE_CHECKING

from modules.adaptive_trend_LTS_mini.cli.config_manager import ConfigManager
from modules.adaptive_trend_LTS_mini.cli.display import (
    display_atc_signals,
    display_manual_mode_config,
)
from modules.adaptive_trend_LTS_mini.cli.input_utils import get_symbol_input
from modules.adaptive_trend_LTS_mini.core.analyzer import analyze_symbol
from modules.common.utils import log_error

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher

__all__ = ["ManualModeExecutor"]


class ManualModeExecutor:
    """
    Executes manual mode analysis for single symbol.

    Handles analyzing a specific symbol and displaying detailed
    ATC signals and trend information.
    """

    def __init__(
        self,
        args: Namespace,
        data_fetcher: "DataFetcher",
        config_manager: ConfigManager,
    ):
        """
        Initialize ManualModeExecutor.

        Args:
            args: Command-line arguments namespace
            data_fetcher: DataFetcher instance for market data
            config_manager: ConfigManager for ATC configuration
        """
        self.args = args
        self.data_fetcher = data_fetcher
        self.config_manager = config_manager

    def analyze_and_display(
        self,
        symbol: str,
        timeframe: str,
    ) -> bool:
        """
        Analyze symbol and display results.

        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe for analysis

        Returns:
            True if analysis succeeded, False otherwise
        """
        # Create ATC config
        atc_config = self.config_manager.create_config(timeframe)

        # Analyze symbol
        result = analyze_symbol(
            symbol=symbol,
            data_fetcher=self.data_fetcher,
            config=atc_config,
        )

        if result is None:
            log_error("Analysis failed")
            return False

        # Display results
        display_atc_signals(
            symbol=result["symbol"],
            df=result["df"],
            atc_results=result["atc_results"],
            current_price=result["current_price"],
            exchange_label=result["exchange_label"],
        )

        return True

    def execute(self, timeframe: str) -> Optional[str]:
        """
        Execute manual mode: get symbol, analyze and display.

        Args:
            timeframe: Timeframe for analysis

        Returns:
            Symbol that was analyzed, or None if failed
        """
        # Get symbol from user input or args
        symbol = get_symbol_input(self.args)

        # Display configuration
        display_manual_mode_config(symbol, timeframe, self.args)

        # Analyze and display
        success = self.analyze_and_display(symbol, timeframe)

        return symbol if success else None
