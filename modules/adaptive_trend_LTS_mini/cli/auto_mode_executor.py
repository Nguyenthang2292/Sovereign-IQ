"""
Auto Mode Execution for ATC Analysis.

This module handles automatic scanning of multiple symbols
and result display.
"""

from argparse import Namespace
from typing import TYPE_CHECKING, List, Optional, Tuple

import pandas as pd

from modules.adaptive_trend_LTS_mini.cli.config_manager import ConfigManager
from modules.adaptive_trend_LTS_mini.cli.display import (
    display_auto_mode_config,
    display_scan_results,
)
from modules.adaptive_trend_LTS_mini.core.scanner import scan_all_symbols

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher

__all__ = ["AutoModeExecutor"]

DEFAULT_EXECUTION_MODE = "threadpool"


class AutoModeExecutor:
    """
    Executes auto mode scanning and displays results.

    Handles scanning all symbols for LONG/SHORT signals and
    displaying the results in a formatted table.
    """

    def __init__(
        self,
        args: Namespace,
        data_fetcher: "DataFetcher",
        config_manager: ConfigManager,
    ):
        """
        Initialize AutoModeExecutor.

        Args:
            args: Command-line arguments namespace
            data_fetcher: DataFetcher instance for market data
            config_manager: ConfigManager for ATC configuration
        """
        self.args = args
        self.data_fetcher = data_fetcher
        self.config_manager = config_manager

    def run_scan(
        self,
        timeframe: str,
        symbols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run ATC auto scan and return results.

        Args:
            timeframe: Timeframe for analysis
            symbols: Optional list of symbols to scan (scans all if None)

        Returns:
            Tuple of (long_signals_df, short_signals_df)
        """
        # Create ATC config
        atc_config = self.config_manager.create_config(timeframe)

        # Scan symbols (provided list or all from exchange)
        long_signals, short_signals = scan_all_symbols(
            data_fetcher=self.data_fetcher,
            atc_config=atc_config,
            max_symbols=self.args.max_symbols,
            min_signal=self.args.min_signal,
            batch_size=getattr(self.args, "batch_size", atc_config.batch_size),
            execution_mode=getattr(self.args, "execution_mode", DEFAULT_EXECUTION_MODE),
            npartitions=getattr(self.args, "npartitions", None),
            symbols=symbols,
        )

        return long_signals, short_signals

    def execute(self, timeframe: str) -> None:
        """
        Execute auto mode: scan and display results.

        Args:
            timeframe: Timeframe for analysis
        """
        # Display configuration
        display_auto_mode_config(timeframe, self.args)

        # Run scan
        long_signals, short_signals = self.run_scan(timeframe)

        # Display results
        display_scan_results(long_signals, short_signals, self.args.min_signal)
