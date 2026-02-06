"""
Interactive Loop for ATC Analysis.

This module handles the interactive prompt loop for analyzing
multiple symbols in sequence.
"""

from argparse import Namespace
from typing import TYPE_CHECKING

from colorama import Fore

from config import DEFAULT_QUOTE
from modules.adaptive_trend_LTS_mini.cli.config_manager import ConfigManager
from modules.adaptive_trend_LTS_mini.cli.manual_mode_executor import ManualModeExecutor
from modules.common.utils import (
    color_text,
    normalize_symbol,
    prompt_user_input,
)

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher

__all__ = ["InteractiveLoop"]


class InteractiveLoop:
    """
    Manages interactive analysis loop.

    Handles continuous symbol analysis with user prompts for
    entering new symbols.
    """

    def __init__(
        self,
        args: Namespace,
        data_fetcher: "DataFetcher",
        config_manager: ConfigManager,
    ):
        """
        Initialize InteractiveLoop.

        Args:
            args: Command-line arguments namespace
            data_fetcher: DataFetcher instance for market data
            config_manager: ConfigManager for ATC configuration
        """
        self.args = args
        self.data_fetcher = data_fetcher
        self.config_manager = config_manager
        self.manual_executor = ManualModeExecutor(
            args=args,
            data_fetcher=data_fetcher,
            config_manager=config_manager,
        )

    def run(self, initial_symbol: str, timeframe: str) -> None:
        """
        Run interactive loop for continuous analysis.

        Args:
            initial_symbol: Starting symbol for the loop
            timeframe: Timeframe for analysis
        """
        symbol = initial_symbol
        quote = self.args.quote.upper() if self.args.quote else DEFAULT_QUOTE

        try:
            while True:
                print(
                    color_text(
                        "\nPress Ctrl+C to exit. Provide a new symbol to continue.",
                        Fore.YELLOW,
                    )
                )
                symbol_input = prompt_user_input(
                    f"Enter symbol pair (default: {symbol}): ",
                    default=symbol,
                )

                symbol = normalize_symbol(symbol_input, quote)

                # Analyze and display (without showing config header again)
                self.manual_executor.analyze_and_display(symbol, timeframe)

        except KeyboardInterrupt:
            print(color_text("\nExiting program by user request.", Fore.YELLOW))
