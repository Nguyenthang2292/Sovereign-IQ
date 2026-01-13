"""Command-line interface (CLI) package for the adaptive_trend module.

This package provides utilities for parsing command-line arguments, displaying analytic
results, and running interactive and main workflows related to adaptive trend analysis.

Exports:
    - parse_args: Argument parser for CLI usage.
    - UserExitRequested: Exception for user-triggered program exit.
    - prompt_timeframe, prompt_interactive_mode: User input prompts for interactive CLI mode.
    - display_atc_signals, display_scan_results, list_futures_symbols: Display and utility functions.
    - main, ATCAnalyzer: Main entrypoint and core analyzer for CLI operation.
"""

from modules.adaptive_trend.cli.argument_parser import parse_args
from modules.adaptive_trend.cli.display import (
    display_atc_signals,
    display_scan_results,
    list_futures_symbols,
)
from modules.adaptive_trend.cli.interactive_prompts import (
    UserExitRequested,
    prompt_interactive_mode,
    prompt_timeframe,
)
from modules.adaptive_trend.cli.main import ATCAnalyzer, main

__all__ = [
    # Argument parsing
    "parse_args",
    # Interactive prompts
    "UserExitRequested",
    "prompt_timeframe",
    "prompt_interactive_mode",
    # Display utilities
    "display_atc_signals",
    "display_scan_results",
    "list_futures_symbols",
    # Main CLI
    "main",
    "ATCAnalyzer",
]
