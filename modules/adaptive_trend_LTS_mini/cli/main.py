"""
Adaptive Trend Classification (ATC) Main Program

Analyzes futures pairs on Binance using Adaptive Trend Classification:
- Fetches OHLCV data from Binance futures
- Calculates ATC signals using multiple moving averages
- Displays trend signals and analysis
"""

import sys
import warnings
from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# Ensure project root is on sys.path when running this file directly (e.g. python -m cli.main)
# Prefer installing the package (pip install -e .) so imports work without path manipulation.
if "__file__" in globals():
    _cli_dir = Path(__file__).resolve().parent
    _project_root = _cli_dir.parent.parent.parent
    _root_str = str(_project_root)
    if _root_str not in sys.path:
        sys.path.insert(0, _root_str)

from modules.common.utils import configure_windows_stdio

# Fix encoding issues on Windows for interactive CLI runs only
configure_windows_stdio()

from colorama import Fore
from colorama import init as colorama_init

from config import (
    DEFAULT_QUOTE,
    DEFAULT_SYMBOL,
)

DEFAULT_EXECUTION_MODE = "threadpool"


from modules.adaptive_trend_LTS_mini.cli.argument_parser import parse_args
from modules.adaptive_trend_LTS_mini.cli.auto_mode_executor import AutoModeExecutor
from modules.adaptive_trend_LTS_mini.cli.config_manager import ConfigManager
from modules.adaptive_trend_LTS_mini.cli.config_utils import ATCParams, get_atc_params
from modules.adaptive_trend_LTS_mini.cli.display import (
    list_futures_symbols,
)
from modules.adaptive_trend_LTS_mini.cli.interactive_loop import InteractiveLoop
from modules.adaptive_trend_LTS_mini.cli.manual_mode_executor import ManualModeExecutor
from modules.adaptive_trend_LTS_mini.cli.mode_manager import ModeManager
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.utils import (
    color_text,
    log_error,
    log_progress,
)

# Suppress only FutureWarning for cleaner CLI output; DeprecationWarning and others remain visible
warnings.filterwarnings("ignore", category=FutureWarning)
colorama_init(autoreset=True)


class ATCAnalyzer:
    """
    ATC Analysis Orchestrator.

    Manages the complete ATC analysis workflow including mode selection,
    configuration, and execution of auto/manual analysis modes.

    This class now acts as a facade, delegating responsibilities to
    specialized components following Single Responsibility Principle.
    """

    def __init__(self, args: Namespace, data_fetcher: DataFetcher):
        """
        Initialize ATC Analyzer.

        Args:
            args: Parsed command-line arguments
            data_fetcher: DataFetcher instance
        """
        self.args = args
        self.data_fetcher = data_fetcher

        # Initialize component managers
        self.mode_manager = ModeManager(args)
        self.config_manager = ConfigManager(args)
        self.auto_executor = AutoModeExecutor(args, data_fetcher, self.config_manager)
        self.manual_executor = ManualModeExecutor(args, data_fetcher, self.config_manager)
        self.interactive_loop = InteractiveLoop(args, data_fetcher, self.config_manager)

        # Keep backward compatibility properties
        self.selected_timeframe = args.timeframe
        self.mode = "manual"
        self._atc_params = None

    def run(self) -> None:
        """Run the analyzer orchestrator."""
        # Determine mode and timeframe
        self.mode, self.selected_timeframe = self.mode_manager.determine_mode_and_timeframe()

        # Run appropriate mode
        if self.mode == "auto":
            self.run_auto_mode()
        else:
            self.run_manual_mode()

    def run_auto_scan(self, symbols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run ATC auto scan and return results without displaying.

        Args:
            symbols: Optional list of symbols to scan.

        Returns:
            Tuple of (long_signals_df, short_signals_df)
        """
        return self.auto_executor.run_scan(
            timeframe=self.selected_timeframe,
            symbols=symbols,
        )

    def run_auto_mode(self) -> None:
        """Run auto mode: scan all symbols for LONG/SHORT signals."""
        self.auto_executor.execute(self.selected_timeframe)

    def run_manual_mode(self) -> None:
        """Run manual mode: analyze specific symbol."""
        # Execute manual mode and get analyzed symbol
        symbol = self.manual_executor.execute(self.selected_timeframe)

        if symbol is None:
            return

        # Interactive loop if prompts enabled
        if not self.args.no_prompt:
            self.interactive_loop.run(
                initial_symbol=symbol,
                timeframe=self.selected_timeframe,
            )

    def run_interactive_loop(self, symbol: str, quote: str, atc_params: dict) -> None:
        """
        Run interactive loop for analyzing multiple symbols.

        DEPRECATED: This method is kept for backward compatibility.
        Use InteractiveLoop.run() instead. Will be removed in a future release.

        Args:
            symbol: Initial symbol
            quote: Quote currency (not used - taken from args)
            atc_params: ATC parameters dictionary (not used - taken from config_manager)
        """
        warnings.warn(
            "run_interactive_loop is deprecated; use InteractiveLoop.run() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.interactive_loop.run(
            initial_symbol=symbol,
            timeframe=self.selected_timeframe,
        )


def initialize_components() -> DataFetcher:
    """
    Initialize and return DataFetcher (contains ExchangeManager).

    Returns:
        DataFetcher instance
    """
    log_progress("Initializing components...")
    exchange_manager = ExchangeManager()
    return DataFetcher(exchange_manager)


def main() -> None:
    """
    Main function for ATC analysis.

    Orchestrates the complete ATC analysis workflow:
    1. Parse command-line arguments
    2. Initialize components (ExchangeManager, DataFetcher)
    3. Create ATC Analyzer instance
    4. Determine mode and timeframe
    5. Run appropriate analysis mode
    """
    args = parse_args()

    # List symbols if requested
    if args.list_symbols:
        data_fetcher = initialize_components()
        list_futures_symbols(data_fetcher)
        return

    # Initialize components
    data_fetcher = initialize_components()

    # Create analyzer instance
    analyzer = ATCAnalyzer(args, data_fetcher)

    # Run analysis
    analyzer.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\nExiting program by user request.", Fore.YELLOW))
        sys.exit(0)
    except Exception as e:
        log_error(f"Error: {type(e).__name__}: {e}")
        import traceback

        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
