"""
Command-line interface components for pairs trading analysis.

This package provides CLI utilities including argument parsing, interactive prompts,
input validation, and formatted display functions.
"""

from modules.pairs_trading.cli.argument_parser import parse_args

# Display utilities
from modules.pairs_trading.cli.display import (
    display_pairs_opportunities,
    display_performers,
)

# Input parsers
from modules.pairs_trading.cli.input_parsers import (
    parse_symbols,
    parse_weights,
    standardize_symbol_input,
)

# Interactive prompts
from modules.pairs_trading.cli.interactive_prompts import (
    prompt_candidate_depth,
    prompt_interactive_mode,
    prompt_kalman_preset_selection,
    prompt_opportunity_preset_selection,
    prompt_target_pairs,
    prompt_weight_preset_selection,
)
from modules.pairs_trading.cli.main import main

__all__ = [
    # Argument parsing
    "parse_args",
    # Interactive prompts
    "prompt_interactive_mode",
    "prompt_weight_preset_selection",
    "prompt_kalman_preset_selection",
    "prompt_opportunity_preset_selection",
    "prompt_target_pairs",
    "prompt_candidate_depth",
    # Input parsers
    "parse_weights",
    "parse_symbols",
    "standardize_symbol_input",
    # Display formatters
    "display_performers",
    "display_pairs_opportunities",
    # Main CLI
    "main",
]
