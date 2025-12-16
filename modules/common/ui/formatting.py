"""
Text formatting and user input utilities.
"""

import pandas as pd
from typing import Optional
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
    user_input = input(color_text(prompt, color)).strip()
    return user_input if user_input else (default or "")


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

