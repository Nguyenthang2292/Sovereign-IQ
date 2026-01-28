"""
Logging functions organized by severity level and purpose.
"""

from colorama import Fore, Style

from modules.common.ui.formatting import color_text


# Standard severity levels
def log_info(msg: str) -> None:
    """Print informational message with blue color."""
    print(color_text(msg, Fore.BLUE))


def log_success(msg: str) -> None:
    """Print success message with green color."""
    print(color_text(msg, Fore.GREEN))


def log_error(msg: str) -> None:
    """Print error message with red color and bright style."""
    print(color_text(msg, Fore.RED, Style.BRIGHT))


def log_warn(msg: str) -> None:
    """Print warning message with yellow color."""
    print(color_text(msg, Fore.YELLOW))


def log_debug(msg: str) -> None:
    """Print debug message with white color."""
    print(color_text(msg, Fore.WHITE))


# Domain-specific logging
def log_data(msg: str) -> None:
    """Print data-related message with cyan color."""
    print(color_text(msg, Fore.CYAN))


def log_analysis(msg: str) -> None:
    """Print analysis-related message with magenta color."""
    print(color_text(msg, Fore.MAGENTA))


def log_model(msg: str) -> None:
    """Print model-related message with magenta color."""
    print(color_text(msg, Fore.MAGENTA))


def log_exchange(msg: str) -> None:
    """Print exchange-related message with cyan color."""
    print(color_text(msg, Fore.CYAN))


def log_system(msg: str) -> None:
    """Print system-level message with white color."""
    print(color_text(msg, Fore.WHITE))


def log_progress(msg: str) -> None:
    """Print progress update message with yellow color."""
    print(color_text(msg, Fore.YELLOW))
