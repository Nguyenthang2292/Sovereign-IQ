"""
Interactive Loop for ATC Analysis.

This module handles the interactive prompt loop for analyzing
multiple symbols in sequence.
"""

from argparse import Namespace
from typing import TYPE_CHECKING

import pandas as pd
from colorama import Fore

from config import DEFAULT_QUOTE
from modules.adaptive_trend_LTS_mini.cli.config_manager import ConfigManager
from modules.adaptive_trend_LTS_mini.cli.display import display_atc_signals
from modules.adaptive_trend_LTS_mini.core.analyzer import analyze_symbol
from modules.common.domain.symbol_codec import SymbolCodec
from modules.common.utils import (
    color_text,
    prompt_user_input,
)

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher

__all__ = ["InteractiveLoop"]

_SYMBOL_CODEC = SymbolCodec()


def prompt_interactive_mode(default_symbol: str) -> str:
    """Prompt for the next symbol in the interactive loop.

    This wrapper is patchable in tests.
    """
    return prompt_user_input(
        f"Enter symbol pair (default: {default_symbol}): ",
        default=default_symbol,
    )


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
                # Analyze and display (without showing config header again)
                atc_config = self.config_manager.create_config(timeframe)
                result = analyze_symbol(
                    symbol=symbol,
                    data_fetcher=self.data_fetcher,
                    config=atc_config,
                )
                if result is not None:
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

                print(
                    color_text(
                        "\nPress Ctrl+C to exit. Provide a new symbol to continue.",
                        Fore.YELLOW,
                    )
                )
                symbol_input = prompt_interactive_mode(symbol)
                if not symbol_input:
                    break

                if "/" not in symbol_input and not symbol_input.upper().endswith(quote):
                    symbol_input = f"{symbol_input}/{quote}"
                symbol = str(_SYMBOL_CODEC.to_ccxt(symbol_input))

        except KeyboardInterrupt:
            print(color_text("\nExiting program by user request.", Fore.YELLOW))
