"""Benchmark comparison package for adaptive_trend modules."""

from .build import ensure_cuda_extensions_built, ensure_rust_extensions_built
from .comparison import compare_signals, generate_comparison_table
from .data import fetch_symbols_data
from .main import main
from .runners import (
    run_cuda_dask_module,
    run_cuda_module,
    run_dask_module,
    run_enhanced_module,
    run_original_module,
    run_rust_batch_module,
    run_rust_cuda_dask_module,
    run_rust_dask_module,
    run_rust_module,
)

__all__ = [
    # Build functions
    "ensure_rust_extensions_built",
    "ensure_cuda_extensions_built",
    # Data functions
    "fetch_symbols_data",
    # Runner functions
    "run_original_module",
    "run_enhanced_module",
    "run_rust_module",
    "run_rust_batch_module",
    "run_cuda_module",
    "run_dask_module",
    "run_rust_dask_module",
    "run_cuda_dask_module",
    "run_rust_cuda_dask_module",
    # Comparison functions
    "compare_signals",
    "generate_comparison_table",
    # Main entry point
    "main",
]
