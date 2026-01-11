
from colorama import Fore, Style

from modules.common.ui.formatting import color_text
from modules.common.ui.formatting import color_text

"""
Logging functions organized by severity level and purpose.
"""




# Standard severity levels
def log_info(message: str) -> None:
    """Print informational message with blue color."""
    print(color_text(message, Fore.BLUE))


def log_success(message: str) -> None:
    """Print success message with green color."""
    print(color_text(message, Fore.GREEN))


def log_error(message: str) -> None:
    """Print error message with red color and bright style."""
    print(color_text(message, Fore.RED, Style.BRIGHT))


def log_warn(message: str) -> None:
    """Print warning message with yellow color."""
    print(color_text(message, Fore.YELLOW))


def log_debug(message: str) -> None:
    """Print debug message with white color."""
    print(color_text(message, Fore.WHITE))


# Domain-specific logging
def log_data(message: str) -> None:
    """Print data-related message with cyan color."""
    print(color_text(message, Fore.CYAN))


def log_analysis(message: str) -> None:
    """Print analysis-related message with magenta color."""
    print(color_text(message, Fore.MAGENTA))


def log_model(message: str) -> None:
    """Print model-related message with magenta color."""
    print(color_text(message, Fore.MAGENTA))


def log_exchange(message: str) -> None:
    """Print exchange-related message with cyan color."""
    print(color_text(message, Fore.CYAN))


def log_system(message: str) -> None:
    """Print system-level message with white color."""
    print(color_text(message, Fore.WHITE))


def log_progress(message: str) -> None:
    """Print progress update message with yellow color."""
    print(color_text(message, Fore.YELLOW))
