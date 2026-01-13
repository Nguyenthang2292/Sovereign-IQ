"""
This module provides interactive CLI prompts for user input, specifically for
configuration of adaptive trend analysis modules. Functions include menu-based
and direct input prompts for selecting analysis timeframes and other
parameters, along with support for input validation and user-friendly messages.

Key functionalities:
- Prompting and validating timeframe selection with clear interactive guidance.
- Handling user exit requests gracefully during input prompts.
- Utilizing colored CLI output for enhanced usability.
"""

from typing import Dict, Optional

from colorama import Fore, Style

from modules.common.utils import (
    color_text,
    log_data,
    log_error,
    log_info,
    log_warn,
    prompt_user_input,
)
from modules.common.utils.domain import TIMEFRAME_NORMALIZED_RE, normalize_timeframe

try:
    from config import DEFAULT_TIMEFRAME
except ImportError:
    DEFAULT_TIMEFRAME = "1h"


class UserExitRequested(Exception):
    """Raised when user requests to exit the application."""

    pass


def prompt_timeframe(default_timeframe: str = DEFAULT_TIMEFRAME) -> str:
    """
    Interactive menu for selecting timeframe.

    Args:
        default_timeframe: Default timeframe to use

    Returns:
        Selected timeframe string in normalized format (e.g., '15m', '1h', '4h', '1d', '1w').
        Custom timeframes are validated to ensure they match the expected format:
        - Number + unit (e.g., '15m', '1h', '4h', '1d', '1w')
        - Unit + number (e.g., 'm15', 'h1', 'd1') - will be normalized
        - Units: m (minutes), h (hours), d (days), w (weeks)
        - Invalid formats will prompt the user to re-enter
    """
    timeframes = [
        ("15m", "15 minutes"),
        ("30m", "30 minutes"),
        ("1h", "1 hour"),
        ("2h", "2 hours"),
        ("4h", "4 hours"),
    ]

    print("\n" + color_text("=" * 60, Fore.CYAN))
    print(color_text("SELECT TIMEFRAME", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 60, Fore.CYAN))

    # Find default index
    default_idx = 0
    for idx, (tf, _) in enumerate(timeframes):
        if tf == default_timeframe:
            default_idx = idx
            break

    for idx, (tf, desc) in enumerate(timeframes, 1):
        if tf == default_timeframe:
            # Highlight default option in magenta (pink)
            option_text = color_text(f"{idx:2d}) {tf:4s} - {desc}", Fore.MAGENTA, Style.BRIGHT)
            print(option_text)
        else:
            print(f"{idx:2d}) {tf:4s} - {desc}")

    print(f"{len(timeframes) + 1:2d}) Custom timeframe")
    print(f"{len(timeframes) + 2:2d}) Use default ({default_timeframe})")

    while True:
        choice = prompt_user_input(
            f"\nSelect timeframe [1-{len(timeframes) + 2}] (default {default_idx + 1}): ",
            default=str(default_idx + 1),
        )

        if not choice:
            return default_timeframe

        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(timeframes):
                return timeframes[choice_num - 1][0]
            elif choice_num == len(timeframes) + 1:
                # Custom timeframe
                while True:
                    custom = prompt_user_input(
                        f"Enter custom timeframe (e.g., 1h, 4h, 1d) [{default_timeframe}]: ",
                        default=default_timeframe,
                    )
                    if not custom:
                        return default_timeframe

                    # Validate timeframe format
                    try:
                        normalized = normalize_timeframe(custom)
                        # Check if normalized format matches expected pattern
                        if TIMEFRAME_NORMALIZED_RE.match(normalized.lower()):
                            return normalized
                        else:
                            log_error(
                                f"Invalid timeframe format: '{custom}'. "
                                "Expected format: number + unit (e.g., '15m', '1h', '4h', '1d', '1w'). "
                                "Units: m (minutes), h (hours), d (days), w (weeks)."
                            )
                    except ValueError as e:
                        log_error(f"Invalid timeframe: {e}")
            elif choice_num == len(timeframes) + 2:
                return default_timeframe
            else:
                log_error(f"Invalid choice. Please enter 1-{len(timeframes) + 2}.")
        except ValueError:
            log_error("Invalid input. Please enter a number.")


def prompt_interactive_mode(default_timeframe: str = DEFAULT_TIMEFRAME) -> Dict[str, Optional[str]]:
    """
    Interactive menu for selecting analysis mode and timeframe.

    Args:
        default_timeframe: Default timeframe to use

    Returns:
        dict with 'mode' key ('auto' or 'manual') and 'timeframe' key.
        When option 3 (timeframe-only) is selected, returns {"mode": None, "timeframe": ...}.
        The caller should handle mode=None by calling this function again to get mode selection.

    Raises:
        UserExitRequested: When user selects option 4 (Exit). The caller should catch
                          this exception and handle the exit appropriately (e.g., sys.exit(0)).
    """
    log_data("=" * 60)
    log_info("Adaptive Trend Classification (ATC) - Interactive Launcher")
    log_data("=" * 60)
    print(
        color_text(
            "1) Auto mode  - scan entire market for LONG/SHORT signals",
            Fore.MAGENTA,
            Style.BRIGHT,
        )
    )
    print("2) Manual mode - analyze specific symbol")
    print("3) Select timeframe")
    print("4) Exit")

    while True:
        choice = prompt_user_input(
            "\nSelect option [1-4] (default 1): ",
            default="1",
        )
        if choice in {"1", "2", "3", "4"}:
            break
        log_error("Invalid choice. Please enter 1, 2, 3, or 4.")

    if choice == "4":
        log_warn("Exiting by user request.")
        raise UserExitRequested()

    if choice == "3":
        # Timeframe selection only
        selected_timeframe = prompt_timeframe(default_timeframe)
        return {"mode": None, "timeframe": selected_timeframe}

    # Ask for timeframe after mode selection
    selected_timeframe = prompt_timeframe(default_timeframe)

    return {"mode": "auto" if choice == "1" else "manual", "timeframe": selected_timeframe}
