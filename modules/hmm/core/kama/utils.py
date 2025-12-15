"""
HMM-KAMA Utility Functions.

This module contains utility functions like decorators and context managers.
"""

from contextlib import contextmanager
import functools
import threading
from typing import Any, List

from modules.common.utils import log_warn, log_error


_thread_local = threading.local()


def prevent_infinite_loop(max_calls=3):
    """Decorator to prevent infinite loops in function calls"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(_thread_local, "call_counts"):
                _thread_local.call_counts = {}

            func_name = func.__name__
            if func_name not in _thread_local.call_counts:
                _thread_local.call_counts[func_name] = 0

            _thread_local.call_counts[func_name] += 1

            try:
                if _thread_local.call_counts[func_name] > 1:
                    log_warn(
                        f"Multiple calls detected for {func_name} ({_thread_local.call_counts[func_name]}). Possible infinite loop."
                    )
                    if _thread_local.call_counts[func_name] > max_calls:
                        log_error(
                            f"Too many recursive calls for {func_name}. Breaking to prevent infinite loop."
                        )
                        # Import here to avoid circular dependency
                        from modules.hmm.core.kama.models import HMM_KAMA
                        # Decrement counter before returning
                        _thread_local.call_counts[func_name] -= 1
                        return HMM_KAMA(-1, -1, -1, -1, -1, -1)

                result = func(*args, **kwargs)
                return result
            finally:
                # Decrement counter instead of resetting to 0
                # This allows detection of recursive calls within the same call stack
                if _thread_local.call_counts[func_name] > 0:
                    _thread_local.call_counts[func_name] -= 1

        return wrapper

    return decorator


@contextmanager
def timeout_context(seconds):
    """Cross-platform timeout context manager
    
    Note: This cannot interrupt a long-running operation, but will detect if it exceeded timeout.
    For true timeout interruption, consider using signal (Unix) or multiprocessing.
    """
    timeout_occurred = threading.Event()
    timer = threading.Timer(seconds, timeout_occurred.set)
    timer.start()

    try:
        yield
        # Check timeout after operation completes
        if timeout_occurred.is_set():
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
    finally:
        timer.cancel()

