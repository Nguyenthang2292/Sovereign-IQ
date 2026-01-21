"""ATC Symbol Scanner.

This sub-module provides functions for scanning multiple symbols using
Adaptive Trend Classification (ATC) to find LONG/SHORT signals.

The scanner fetches data for multiple symbols, calculates ATC signals,
and filters results based on signal strength and trend direction.
"""

from __future__ import annotations

from .scan_all_symbols import scan_all_symbols

__all__ = [
    "scan_all_symbols",
]
