"""
Logging functions organized by severity level and purpose.

All functions support optional printf-style positional arguments:
    log_info("fetched %s candles for %s", count, symbol)
"""

from colorama import Fore, Style

from modules.common.ui.formatting import color_text


def _fmt(msg: str, args: tuple) -> str:
    """Format message with optional printf-style args."""
    if args:
        try:
            return msg % args
        except (TypeError, ValueError):
            # Fallback: append args as-is to avoid crashing
            return f"{msg} {' '.join(str(a) for a in args)}"
    return msg


# Standard severity levels
def log_info(msg: str, *args: object) -> None:
    """Print informational message with blue color."""
    print(color_text(_fmt(msg, args), Fore.BLUE))


def log_success(msg: str, *args: object) -> None:
    """Print success message with green color."""
    print(color_text(_fmt(msg, args), Fore.GREEN))


def log_error(msg: str, *args: object, exc_info: bool = False) -> None:
    """Print error message with red color and bright style."""
    if exc_info:
        import traceback

        traceback.print_exc()
    print(color_text(_fmt(msg, args), Fore.RED, Style.BRIGHT))


def log_warn(msg: str, *args: object) -> None:
    """Print warning message with yellow color."""
    print(color_text(_fmt(msg, args), Fore.YELLOW))


def log_debug(msg: str, *args: object) -> None:
    """Print debug message with white color."""
    print(color_text(_fmt(msg, args), Fore.WHITE))


# Domain-specific logging
def log_data(msg: str, *args: object) -> None:
    """Print data-related message with cyan color."""
    print(color_text(_fmt(msg, args), Fore.CYAN))


def log_analysis(msg: str, *args: object) -> None:
    """Print analysis-related message with magenta color."""
    print(color_text(_fmt(msg, args), Fore.MAGENTA))


def log_model(msg: str, *args: object) -> None:
    """Print model-related message with magenta color."""
    print(color_text(_fmt(msg, args), Fore.MAGENTA))


def log_exchange(msg: str, *args: object) -> None:
    """Print exchange-related message with cyan color."""
    print(color_text(_fmt(msg, args), Fore.CYAN))


def log_system(msg: str, *args: object) -> None:
    """Print system-level message with white color."""
    print(color_text(_fmt(msg, args), Fore.WHITE))


def log_progress(msg: str, *args: object) -> None:
    """Print progress update message with yellow color."""
    print(color_text(_fmt(msg, args), Fore.YELLOW))


def log_memory(threshold_mb: int = 1000) -> None:
    """
    Log memory usage if it exceeds threshold.
    Requires psutil.
    """
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / 1024 / 1024

        if rss_mb > threshold_mb:
            print(color_text(f"MEMORY WARNING: Usage {rss_mb:.2f} MB", Fore.RED, Style.BRIGHT))
        # else:
        #     # Optional: log normal memory usage as debug
        #     # print(color_text(f"Memory: {rss_mb:.2f} MB", Fore.WHITE))
    except ImportError:
        pass
