"""
Equity calculations for Adaptive Trend Classification (ATC).

This package provides functions to calculate equity curves based on trading
signals and returns. The equity curve simulates performance of a trading
strategy using exponential growth factors and decay rates.

Performance optimization:
- Uses Numba JIT compilation for the core equity calculation loop
- Replaces pd.NA with np.nan for better compatibility with float64 dtype

Modules:
- core: Core calculation functions (_calculate_equity_core, _calculate_equity_vectorized)
- equity_series: Main public API (equity_series function)
- utils: Utility functions and Numba support
"""

from __future__ import annotations

from .core import _calculate_equity_core, _calculate_equity_vectorized
from .equity_series import equity_series

__all__ = [
    "equity_series",
    "_calculate_equity_vectorized",
    "_calculate_equity_core",
]
