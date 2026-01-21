"""
Utility functions and imports for equity calculations.

This module provides:
- Numba JIT compilation support with fallback
- Common imports used across equity calculation modules
"""

from __future__ import annotations

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    # Fallback if numba is not installed
    def njit(*args, **kwargs):
        """No-op decorator when numba is not available."""

        def decorator(func):
            return func

        return decorator

from modules.common.utils import log_warn

if not _HAS_NUMBA:
    try:
        log_warn("Numba not installed. Performance will be degraded. Please install numba.")
    except Exception:
        print("[WARN] Numba not installed. Performance will be degraded.")

__all__ = ["njit", "_HAS_NUMBA"]
