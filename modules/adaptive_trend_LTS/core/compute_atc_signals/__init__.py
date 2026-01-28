"""Adaptive Trend Classification (ATC) - Main computation module.

This sub-module provides the complete ATC signal computation pipeline:

Public API:
- compute_atc_signals: Main orchestration function for ATC computation
- calculate_layer2_equities: Layer 2 equity calculations
- process_symbols_batch_dask: Dask-based batch processing for large datasets (Phase 5)
- IncrementalATC: Incremental ATC computation for live trading (Phase 6)

Internal modules:
- validation: Input validation utilities
- average_signal: Final Average_Signal calculation
- dask_batch_processor: Dask-based out-of-core processing
- incremental_atc: Incremental ATC state management
"""

from .calculate_layer2_equities import calculate_layer2_equities
from .compute_atc_signals import compute_atc_signals
from .dask_batch_processor import process_symbols_batch_dask
from .incremental_atc import IncrementalATC

__all__ = [
    "compute_atc_signals",
    "calculate_layer2_equities",
    "process_symbols_batch_dask",  # Phase 5: Dask support
    "IncrementalATC",  # Phase 6: Incremental ATC
]
