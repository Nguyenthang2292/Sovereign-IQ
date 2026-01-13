"""
Text formatting and user input utilities.
"""

import logging
import sys
from typing import Optional

import pandas as pd
from colorama import Fore, Style


def color_text(text: str, color: str = Fore.WHITE, style: str = Style.NORMAL) -> str:
    """
    Applies color and style to text using colorama.

    Args:
        text: Text to format
        color: Colorama Fore color (default: Fore.WHITE)
        style: Colorama Style (default: Style.NORMAL)

    Returns:
        Formatted text string with color and style codes
    """
    return f"{style}{color}{text}{Style.RESET_ALL}"


def format_price(value: float) -> str:
    """
    Formats prices/indicators with adaptive precision so tiny values remain readable.

    Args:
        value: Numeric value to format

    Returns:
        Formatted price string with appropriate precision, or "N/A" if invalid
    """
    if value is None or pd.isna(value):
        return "N/A"

    abs_val = abs(value)
    if abs_val >= 1:
        precision = 2
    elif abs_val >= 0.01:
        precision = 4
    elif abs_val >= 0.0001:
        precision = 6
    else:
        precision = 8

    return f"{value:.{precision}f}"


def prompt_user_input(
    prompt: str,
    default: Optional[str] = None,
    color: str = Fore.YELLOW,
) -> str:
    """
    Prompt user for input with optional default value and colored prompt.

    Args:
        prompt: Prompt message to display
        default: Default value if user enters empty string
        color: Colorama Fore color for prompt (default: Fore.YELLOW)

    Returns:
        User input string, or default if empty input provided
    """
    from modules.common.utils import safe_input

    user_input = safe_input(color_text(prompt, color), default=default or "")
    return user_input if user_input else (default or "")


def _prompt_with_sentinel(prompt: str, default: Optional[str], color: str, back_sentinel: str = "-") -> tuple:
    """
    Helper function for sentinel-based input logic.

    Builds enhanced prompt with sentinel instruction, reads input, and returns
    appropriate tuple based on input value.

    Args:
        prompt: Prompt message to display
        default: Default value if user enters empty string
        color: Colorama Fore color for prompt
        back_sentinel: Sentinel value to indicate back navigation (default: "-")

    Returns:
        Tuple of (user_input, is_back) where:
        - (None, True) if sentinel was entered
        - (default or "", False) if input was empty
        - (user_input, False) otherwise
    """
    from modules.common.utils import safe_input

    enhanced_prompt = f"{prompt} (enter '{back_sentinel}' to go back): "
    user_input = safe_input(color_text(enhanced_prompt, color), default=default or "")

    if user_input == back_sentinel:
        return (None, True)
    elif user_input == "":
        return (default or "", False)
    else:
        return (user_input, False)


def prompt_user_input_with_backspace(
    prompt: str,
    default: Optional[str] = None,
    color: str = Fore.YELLOW,
) -> tuple:
    """
    Prompt user for input with backspace key detection for back navigation.

    On Windows, detects backspace key (ASCII 8) to return special back signal.
    On other platforms, uses "-" as explicit sentinel to indicate back navigation.
    Empty input uses the default value (matching Windows behavior).

    Args:
        prompt: Prompt message to display
        default: Default value if user enters empty string
        color: Colorama Fore color for prompt (default: Fore.YELLOW)

    Returns:
        Tuple of (user_input, is_back) where is_back is True if back was signaled
    """
    if sys.platform == "win32":
        # Windows: Use msvcrt.getch() to detect backspace
        try:
            import msvcrt
        except ImportError:
            # Fallback to standard input if msvcrt is not available
            return _prompt_with_sentinel(prompt, default, color)

        print(color_text(prompt, color), end="", flush=True)

        result = []
        while True:
            char = msvcrt.getch()

            # Handle backspace (ASCII 8) or Ctrl+H
            if char == b"\x08" or char == b"\x7f":
                if result:
                    result.pop()
                    # Move cursor back, print space, move cursor back again
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                else:
                    # Backspace when input is empty = back signal
                    print()  # New line
                    return (None, True)
                continue

            # Handle Enter (carriage return or newline)
            if char == b"\r" or char == b"\n":
                print()  # New line
                user_input = "".join(result).strip()

                # Empty input means use default, not back
                return (user_input if user_input else (default or ""), False)

            # Handle regular characters
            if isinstance(char, bytes):
                # Use errors='replace' to handle invalid UTF-8 bytes gracefully
                # This replaces invalid bytes with replacement character (U+FFFD) instead of raising exception
                decoded = char.decode("utf-8", errors="replace")
                # Filter out replacement characters to avoid displaying them
                if decoded and decoded != "\ufffd":
                    result.append(decoded)
                    sys.stdout.write(decoded)
                    sys.stdout.flush()
                else:
                    # Log ignored replacement characters
                    logging.debug(f"Ignored invalid UTF-8 byte (replacement character): {char}")
    else:
        # Non-Windows or msvcrt not available: Use standard input with sentinel
        # Use "-" as explicit sentinel for back navigation
        # Empty input uses default value (matching Windows behavior)
        return _prompt_with_sentinel(prompt, default, color)


def extract_dict_from_namespace(namespace, keys: list) -> dict:
    """
    Extract a dictionary from a namespace object using specified keys.

    Args:
        namespace: Namespace object (e.g., from argparse)
        keys: List of attribute names to extract

    Returns:
        Dictionary with extracted key-value pairs
    """
    return {key: getattr(namespace, key, None) for key in keys}
