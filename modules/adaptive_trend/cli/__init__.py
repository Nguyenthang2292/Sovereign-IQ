
from modules.adaptive_trend.cli.argument_parser import parse_args
from modules.adaptive_trend.cli.display import (

"""
Command-line interface components for ATC analysis.

This package provides CLI utilities including argument parsing, interactive prompts,
and formatted display functions.
"""

# Argument parsing

# Display utilities
from modules.adaptive_trend.cli.display import (
    display_atc_signals,
    display_scan_results,
    list_futures_symbols,
)

# Interactive prompts
from modules.adaptive_trend.cli.interactive_prompts import (
    prompt_interactive_mode,
    prompt_timeframe,
)
from modules.adaptive_trend.cli.main import ATCAnalyzer, main

__all__ = [
    # Argument parsing
    "parse_args",
    # Interactive prompts
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
