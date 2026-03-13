"""Incremental ATC computation for live trading.

This module acts as a facade for the refactored `incremental` package.
Original logic has been moved to `modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental`.
"""

from __future__ import annotations

from .incremental import (
    TF_RESOLUTION_MAP,
    IncrementalATC,
    MultiTimeframeIncrementalATC,
)

__all__ = ["IncrementalATC", "MultiTimeframeIncrementalATC", "TF_RESOLUTION_MAP"]
