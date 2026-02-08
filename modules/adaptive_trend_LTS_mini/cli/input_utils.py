"""
Input utilities for ATC CLI.
"""

import sys
from argparse import Namespace
from typing import Tuple

from config import (
    DEFAULT_QUOTE,
    DEFAULT_SYMBOL,
)
from modules.adaptive_trend_LTS_mini.cli.interactive_prompts import (
    UserExitRequested,
    prompt_interactive_mode,
)
from modules.common.utils import (
    log_warn,
    normalize_symbol,
    prompt_user_input,
)


def determine_mode_and_timeframe(args: Namespace) -> Tuple[str, str]:
    """
    Determine analysis mode and timeframe from arguments and interactive menu.

    Args:
        args: Parsed command-line arguments

    Returns:
        tuple: (mode, selected_timeframe)
    """
    selected_timeframe = args.timeframe
    mode = "manual"

    if args.auto:
        mode = "auto"
    elif not args.no_menu and not args.no_prompt and args.symbol is None:
        try:
            menu_result = prompt_interactive_mode(default_tf=args.timeframe)

            # If user only selected timeframe, keep default manual mode
            if "timeframe" in menu_result and "mode" not in menu_result:
                selected_timeframe = str(menu_result.get("timeframe") or args.timeframe)
                mode = "manual"
            else:
                mode = str(menu_result.get("mode") or "manual")
                if "timeframe" in menu_result:
                    selected_timeframe = str(menu_result.get("timeframe") or args.timeframe)
        except UserExitRequested:
            log_warn("Exiting by user request.")
            sys.exit(0)

    return mode, selected_timeframe


def get_symbol_input(args: Namespace) -> str:
    """
    Get symbol input from arguments or user prompt.

    Args:
        args: Parsed command-line arguments

    Returns:
        str: Normalized symbol
    """
    quote = args.quote.upper() if args.quote else DEFAULT_QUOTE
    symbol_input = args.symbol

    if not symbol_input and not args.no_prompt:
        symbol_input = prompt_user_input(
            f"Enter symbol pair (default: {DEFAULT_SYMBOL}): ",
            default=DEFAULT_SYMBOL,
        )

    if not symbol_input:
        symbol_input = DEFAULT_SYMBOL

    # Security validation: Prevent injection
    if not all(c.isalnum() or c in "/-" for c in symbol_input):
        log_warn(f"Invalid characters in symbol input: {symbol_input}. Using default.")
        symbol_input = DEFAULT_SYMBOL

    return normalize_symbol(symbol_input, quote)
