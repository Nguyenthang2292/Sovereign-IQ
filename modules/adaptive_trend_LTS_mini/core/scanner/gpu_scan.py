"""
GPU batch scanning stub for ATC scanner.

This is a minimal stub that falls back to sequential scanning
when GPU support is not available in this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig
    from modules.common.core.data_fetcher import DataFetcher

from modules.common.ui.logging import log_warn

from .sequential import _scan_sequential


def _scan_gpu_batch(
    symbols: List[str],
    data_fetcher: "DataFetcher",
    atc_config: "ATCConfig",
    min_signal: float,
    batch_size: int = 100,
) -> Tuple[list, int, int, list]:
    """
    GPU batch scanning stub - falls back to sequential scanning.

    GPU scanning is not available in the LTS_mini module.
    This function falls back to sequential processing.

    Args:
        symbols: List of symbol strings to scan
        data_fetcher: DataFetcher instance
        atc_config: ATC configuration
        min_signal: Minimum signal threshold
        batch_size: Batch size (unused in fallback)

    Returns:
        Tuple of (results, skipped_count, error_count, skipped_symbols)
    """
    log_warn("GPU scanning not available in LTS_mini, falling back to sequential")
    return _scan_sequential(symbols, data_fetcher, atc_config, min_signal, batch_size)
