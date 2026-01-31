"""Signal generation functions for Adaptive Trend Classification (ATC).

This sub-module provides functions to generate trading signals from Moving Averages:
- crossover: Detect upward crossover (series_a crosses above series_b)
- crossunder: Detect downward crossover (series_a crosses below series_b)
- generate_signal_from_ma: Generate discrete signals {-1, 0, 1} from price/MA crossovers
"""

from __future__ import annotations

from .crossover import crossover
from .crossunder import crossunder
from .generate_signal import generate_signal_from_ma

__all__ = [
    "crossover",
    "crossunder",
    "generate_signal_from_ma",
]
