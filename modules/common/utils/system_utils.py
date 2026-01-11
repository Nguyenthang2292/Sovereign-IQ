
from typing import Optional
import sys

"""
System utilities for cross-platform compatibility.

This module provides system-level utility functions that need
to work consistently across Windows and Unix-like systems.
"""



def setup_windows_stdin() -> None:
    """
    Setup stdin for Windows CLI applications.

    On Windows, stdin may be None when running in certain environments.
    This function attempts to open a console handle (CON) if stdin is missing
    or closed. This is critical before importing modules that might
    close stdin (like ccxt/exchange managers).

    Note:
        This function is safe to call on non-Windows systems where it does nothing.
    """
    if sys.platform == "win32":
        try:
            if sys.stdin is None or (hasattr(sys.stdin, "closed") and sys.stdin.closed):
                sys.stdin = open("CON", "r", encoding="utf-8", errors="replace")
        except (OSError, IOError, AttributeError):
            # If we can't setup stdin, continue anyway
            # This may happen in non-console environments
            pass


def get_error_code(exception: Exception) -> Optional[int]:
    """
    Extract error code from an exception in a safe way.

    Args:
        exception: Exception to extract error code from

    Returns:
        Error code if available, None otherwise
    """
    # Try different attribute names that might contain error codes
    for attr in ["status_code", "code", "errno"]:
        error_code = getattr(exception, attr, None)
        if error_code is not None:
            try:
                return int(error_code)
            except (ValueError, TypeError):
                pass
    return None


def is_retryable_error(exception: Exception) -> bool:
    """
    Check if an exception represents a retryable error.

    Args:
        exception: Exception to check

    Returns:
        True if the error is likely retryable (network, rate limit, etc.)
    """
    error_code = get_error_code(exception)
    error_message = str(exception).lower()

    # Common HTTP status codes for retryable errors
    if error_code in [429, 500, 502, 503, 504]:
        return True

    # Check error message for retryable conditions
    retryable_keywords = ["timeout", "network", "rate limit", "connection refused", "temporary failure"]

    return any(keyword in error_message for keyword in retryable_keywords)
