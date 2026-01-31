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

from typing import Optional, TypedDict

from colorama import Fore, Style

from modules.common.domain.timeframes import (
    TIMEFRAME_NORMALIZED_RE,
    normalize_timeframe,
)
from modules.common.utils import (
    color_text,
    log_data,
    log_error,
    log_info,
    log_warn,
    prompt_user_input,
)

try:
    from config import DEFAULT_TIMEFRAME
except ImportError:
    DEFAULT_TIMEFRAME = "1h"

# Display formatting constants
PROMPT_DISPLAY_WIDTH = 60
MAX_INPUT_LENGTH = 100
MAX_INPUT_ATTEMPTS = 10

# Color constants
MENU_COLOR = Fore.CYAN
HIGHLIGHT_COLOR = Fore.MAGENTA
ERROR_COLOR = Fore.RED


class InteractiveModeResult(TypedDict):
    """Result from interactive mode prompt."""

    mode: Optional[str]
    timeframe: str


class UserExitRequested(Exception):
    """Raised when user requests to exit the application."""


def _validate_input_length(user_input: str, max_length: int = MAX_INPUT_LENGTH) -> bool:
    """Validate input length to prevent memory exhaustion."""
    if len(user_input) > max_length:
        log_error(f"Input too long. Max {max_length} chars.")
        return False
    return True


def _find_timeframe_index(timeframes: list[tuple[str, str]], target: str) -> int:
    """Find index of timeframe in list."""
    for idx, (tf, _) in enumerate(timeframes):
        if tf == target:
            return idx
    return -1


def _prompt_custom_timeframe(default_timeframe: str) -> str:
    """Prompt for custom timeframe with validation."""
    attempts = 0
    while attempts < MAX_INPUT_ATTEMPTS:
        attempts += 1
        custom = prompt_user_input(
            f"Enter custom timeframe [{default_timeframe}]: ",
            default=default_timeframe,
        )
        if not custom:
            return default_timeframe

        if not _validate_input_length(custom):
            continue

        try:
            normalized = normalize_timeframe(custom)
            if TIMEFRAME_NORMALIZED_RE.match(normalized.lower()):
                return normalized
            log_error(f"Format error: '{custom}'. Use '1h', '4h', etc.")
        except ValueError as e:
            log_error(f"Invalid timeframe: {e}")

    log_warn(f"Max retries. Using default '{default_timeframe}'.")
    return default_timeframe


def _display_timeframe_menu(timeframes: list[tuple[str, str]], default_timeframe: str, default_idx: int) -> None:
    """Display timeframe selection menu."""
    print("\n" + color_text("=" * PROMPT_DISPLAY_WIDTH, MENU_COLOR))
    print(color_text("SELECT TIMEFRAME", MENU_COLOR, Style.BRIGHT))
    print(color_text("=" * PROMPT_DISPLAY_WIDTH, MENU_COLOR))

    for idx, (tf, desc) in enumerate(timeframes, 1):
        if tf == default_timeframe:
            # Highlight default option
            option_text = color_text(f"{idx:2d}) {tf:4s} - {desc}", HIGHLIGHT_COLOR, Style.BRIGHT)
            print(option_text)
        else:
            print(f"{idx:2d}) {tf:4s} - {desc}")

    print(f"{len(timeframes) + 1:2d}) Custom timeframe")
    print(f"{len(timeframes) + 2:2d}) Use default ({default_timeframe})")


def prompt_timeframe(default_timeframe: str = DEFAULT_TIMEFRAME) -> str:
    """Interactive menu for selecting timeframe."""
    timeframes: list[tuple[str, str]] = [
        ("15m", "15 minutes"),
        ("30m", "30 minutes"),
        ("1h", "1 hour"),
        ("2h", "2 hours"),
        ("4h", "4 hours"),
    ]

    num_tf = len(timeframes)
    custom_opt = num_tf + 1
    def_opt = num_tf + 2

    d_idx = _find_timeframe_index(timeframes, default_timeframe)
    disp_idx = d_idx + 1 if d_idx != -1 else 1

    _display_timeframe_menu(timeframes, default_timeframe, d_idx)

    attempts = 0
    while attempts < MAX_INPUT_ATTEMPTS:
        attempts += 1
        prompt = f"\nSelect [1-{def_opt}] (default {disp_idx}): "
        choice = prompt_user_input(prompt, default=str(disp_idx))

        if not choice or not _validate_input_length(choice):
            continue

        choice = choice.strip()
        if not choice.isdigit():
            log_error("Enter a number.", color=ERROR_COLOR)
            continue

        c_num = int(choice)
        if 1 <= c_num <= num_tf:
            return timeframes[c_num - 1][0]
        if c_num == custom_opt:
            return _prompt_custom_timeframe(default_timeframe)
        if c_num == def_opt:
            return default_timeframe
        log_error(f"Enter 1-{def_opt}.", color=ERROR_COLOR)

    log_warn(f"Max retries. Using default '{default_timeframe}'.")
    return default_timeframe


def prompt_interactive_mode(
    default_tf: str = DEFAULT_TIMEFRAME,
) -> InteractiveModeResult:
    """Interactive menu for selecting mode and timeframe."""
    log_data("=" * PROMPT_DISPLAY_WIDTH)
    log_info("ATC - Interactive Launcher")
    log_data("=" * PROMPT_DISPLAY_WIDTH)
    print(color_text("1) Auto mode", HIGHLIGHT_COLOR, Style.BRIGHT))
    print("2) Manual mode")
    print("3) Select timeframe")
    print("4) Exit")

    attempts = 0
    while attempts < MAX_INPUT_ATTEMPTS:
        attempts += 1
        p_msg = "\nSelect [1-4] (default 1): "
        choice = prompt_user_input(p_msg, default="1")
        if not _validate_input_length(choice):
            continue

        choice = choice.strip()
        if choice in {"1", "2", "3", "4"}:
            break
        log_error("Enter 1, 2, 3, or 4.", color=ERROR_COLOR)
    else:
        log_warn("Max retries. Exiting.")
        raise UserExitRequested()

    if choice == "4":
        log_warn("Exiting.")
        raise UserExitRequested()

    res_tf = prompt_timeframe(default_tf)
    if choice == "3":
        return {"mode": None, "timeframe": res_tf}

    return {"mode": "auto" if choice == "1" else "manual", "timeframe": res_tf}
