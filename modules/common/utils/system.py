"""
System utilities for platform-specific configuration.
"""

import sys
import io
import os


def configure_windows_stdio() -> None:
    """
    Configure Windows stdio encoding for UTF-8 support.
    
    Only applies to interactive CLI runs, not during pytest.
    This function fixes encoding issues on Windows by wrapping
    stdout and stderr with UTF-8 encoding.
    
    Note:
        - Only runs on Windows (win32 platform)
        - Skips configuration during pytest runs
        - Only configures if stdout/stderr have buffer attribute
    """
    if sys.platform != "win32":
        return
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    if not hasattr(sys.stdout, "buffer") or isinstance(sys.stdout, io.TextIOWrapper):
        return
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

