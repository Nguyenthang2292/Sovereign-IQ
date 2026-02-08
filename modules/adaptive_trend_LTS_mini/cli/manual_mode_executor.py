"""
Manual Mode Execution for ATC Analysis.

This module handles single symbol analysis and result display.
"""

from argparse import Namespace
from typing import TYPE_CHECKING, Optional

import pandas as pd

from config import DEFAULT_QUOTE, DEFAULT_SYMBOL
from modules.adaptive_trend_LTS_mini.cli.config_manager import ConfigManager
from modules.adaptive_trend_LTS_mini.cli.display import (
    display_atc_signals,
    display_manual_mode_config,
)
from modules.adaptive_trend_LTS_mini.core.analyzer import analyze_symbol
from modules.common.utils import log_error, normalize_symbol, prompt_user_input

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher

__all__ = ["ManualModeExecutor"]


def prompt_interactive_mode(default_symbol: str) -> str:
    """Prompt for a symbol in manual mode.

    This thin wrapper is patchable in tests.
    """
    return prompt_user_input(
        f"Enter symbol pair (default: {default_symbol}): ",
        default=default_symbol,
    )


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

        atc_results = result.get("atc_results", {})
        avg_signal = atc_results.get("Average_Signal") if isinstance(atc_results, dict) else None
        if isinstance(avg_signal, pd.Series):
            display_atc_signals(
                symbol=result.get("symbol", symbol),
                df=result.get("df"),
                atc_results=atc_results,
                current_price=result.get("current_price", 0.0),
                exchange_label=result.get("exchange_label", ""),
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
        symbol = self.args.symbol
        if not symbol and not self.args.no_prompt:
            symbol = prompt_interactive_mode(DEFAULT_SYMBOL)
        if not symbol:
            symbol = DEFAULT_SYMBOL

        quote = self.args.quote.upper() if self.args.quote else DEFAULT_QUOTE
        symbol = normalize_symbol(symbol, quote)

        # Display configuration
        display_manual_mode_config(symbol, timeframe, self.args)

        # Analyze and display
        success = self.analyze_and_display(symbol, timeframe)

        return symbol if success else None
