"""
Windows stdin Protection Utility

Context manager to protect and restore stdin on Windows, preventing
"I/O operation on closed file" errors caused by Google SDK initialization.
"""

import contextlib
import sys


@contextlib.contextmanager
def protect_stdin_windows():
    """
    Context manager to protect and restore stdin on Windows.

    This addresses a specific issue where Google SDK initialization (used by GeminiBatchChartAnalyzer)
    may close or interfere with sys.stdin on Windows, causing "I/O operation on closed file" errors
    when the application tries to read user input later.

    The 'CON' device is Windows' console input device. Opening it provides a fallback stdin
    when the original stdin has been closed by external libraries.

    Note: We don't save a reference to the original stdin because if it gets closed during the
    protected operation, both sys.stdin and the saved reference will point to the same closed
    object. Instead, we always open a fresh 'CON' handle for restoration.

    Yields:
        None

    Example:
        ```python
        with protect_stdin_windows():
            analyzer = GeminiBatchChartAnalyzer(...)
        ```
    """
    # Only protect stdin on Windows (other platforms don't have this issue)
    if sys.platform == "win32":
        # If stdin is already closed, reopen it before proceeding
        if sys.stdin is None or (hasattr(sys.stdin, "closed") and sys.stdin.closed):
            try:
                sys.stdin = open("CON", "r", encoding="utf-8", errors="replace")
            except (OSError, IOError):
                # If we can't reopen stdin, continue anyway (non-critical)
                # This is best-effort protection
                pass

    try:
        yield
    finally:
        # Always restore stdin after the protected operation completes
        # Open a fresh 'CON' handle instead of trying to restore a potentially closed reference
        if sys.platform == "win32":
            # If stdin was closed during the operation, reopen it
            if sys.stdin is None or (hasattr(sys.stdin, "closed") and sys.stdin.closed):
                try:
                    sys.stdin = open("CON", "r", encoding="utf-8", errors="replace")
                except (OSError, IOError):
                    # If we can't restore stdin, continue anyway (non-critical)
                    # This is best-effort restoration
                    pass
