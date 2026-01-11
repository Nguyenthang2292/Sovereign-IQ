
from typing import Any, Dict, List, Optional
import pickle

"""
Parallel processing workers for backtester.

This module contains worker functions for multiprocessing,
used for parallel signal calculation.
"""


# Try to import shared memory utilities
try:
    from .shared_memory_utils import (
        SHARED_MEMORY_AVAILABLE,
        reconstruct_dataframe_from_shared_memory,
    )
except ImportError:
    SHARED_MEMORY_AVAILABLE = False

    def reconstruct_dataframe_from_shared_memory(shm_info: Any) -> Any:
        raise RuntimeError("Shared memory utilities not available")


def calculate_signal_batch_worker(
    start_idx: int,
    end_idx: int,
    df_bytes: Optional[bytes] = None,
    shm_info: Optional[Dict[str, Any]] = None,
    symbol: str = None,
    timeframe: str = None,
    limit: int = None,
    signal_type: str = None,
    osc_length: int = None,
    osc_mult: float = None,
    osc_strategies: Optional[List[int]] = None,
    spc_params: Optional[Dict] = None,
    enabled_indicators: Optional[List[str]] = None,
    use_confidence_weighting: bool = True,
    min_indicators_agreement: int = 3,
    **kwargs,  # Accept additional kwargs for compatibility
) -> Dict[int, int]:
    """
    Worker function for parallel signal calculation.

    This function is called by multiprocessing Pool to calculate signals
    for a batch of periods. Supports both pickled DataFrame (df_bytes) and
    shared memory (shm_info) for data passing.

    Args:
        start_idx: Start index of the batch
        end_idx: End index of the batch (exclusive)
        df_bytes: Pickled DataFrame bytes (if using pickle)
        shm_info: Shared memory info dict (if using shared memory)
        Other args: Configuration for signal calculation

    Returns:
        Dictionary mapping period index to signal value
    """
    # Import here to avoid circular import
    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.core.exchange_manager import ExchangeManager
    from modules.common.utils import log_warn
    from modules.position_sizing.core.hybrid_signal_calculator import HybridSignalCalculator

    # Reconstruct DataFrame from either pickle or shared memory
    if shm_info is not None and SHARED_MEMORY_AVAILABLE:
        try:
            df = reconstruct_dataframe_from_shared_memory(shm_info)
        except Exception as e:
            log_warn(f"Failed to reconstruct DataFrame from shared memory: {e}")
            # Fallback to pickle if available
            if df_bytes is not None:
                df = pickle.loads(df_bytes)
            else:
                raise
    elif df_bytes is not None:
        # Unpickle DataFrame
        df = pickle.loads(df_bytes)
    else:
        raise ValueError("Either df_bytes or shm_info must be provided")

    # Create new DataFetcher instance for this worker
    # (DataFetcher contains connection objects that can't be pickled)
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)

    # Create a new HybridSignalCalculator instance for this worker
    hybrid_calc = HybridSignalCalculator(
        data_fetcher=data_fetcher,
        enabled_indicators=enabled_indicators or [],
        use_confidence_weighting=use_confidence_weighting,
        min_indicators_agreement=min_indicators_agreement,
    )

    batch_signals = {}

    # Calculate signals for each period in this batch
    for i in range(start_idx, end_idx):
        try:
            signal, confidence = hybrid_calc.calculate_hybrid_signal(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                period_index=i,
                signal_type=signal_type,
                osc_length=osc_length,
                osc_mult=osc_mult,
                osc_strategies=osc_strategies,
                spc_params=spc_params,
            )
            batch_signals[i] = signal
        except Exception as e:
            log_warn(f"Error calculating signal for period {i} in batch [{start_idx}:{end_idx}]: {e}")
            batch_signals[i] = 0  # Default to no signal on error

    return batch_signals


def calculate_single_signal_batch_worker(
    start_idx: int,
    end_idx: int,
    df_bytes: Optional[bytes] = None,
    shm_info: Optional[Dict[str, Any]] = None,
    symbol: str = None,
    timeframe: str = None,
    limit: int = None,
    osc_length: int = None,
    osc_mult: float = None,
    osc_strategies: Optional[List[int]] = None,
    spc_params: Optional[Dict] = None,
    enabled_indicators: Optional[List[str]] = None,
    use_confidence_weighting: bool = True,
    min_indicators_agreement: int = 3,
    **kwargs,  # Accept additional kwargs for compatibility
) -> Dict[int, int]:
    """
    Worker function for parallel single signal calculation (highest confidence).

    This function is called by multiprocessing Pool to calculate signals
    for a batch of periods using single signal (highest confidence) approach.
    Supports both pickled DataFrame (df_bytes) and shared memory (shm_info) for data passing.

    Args:
        start_idx: Start index of the batch
        end_idx: End index of the batch (exclusive)
        df_bytes: Pickled DataFrame bytes (if using pickle)
        shm_info: Shared memory info dict (if using shared memory)
        Other args: Configuration for signal calculation

    Returns:
        Dictionary mapping period index to signal value
    """
    # Import here to avoid circular import
    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.core.exchange_manager import ExchangeManager
    from modules.common.utils import log_warn
    from modules.position_sizing.core.hybrid_signal_calculator import HybridSignalCalculator

    # Reconstruct DataFrame from either pickle or shared memory
    if shm_info is not None and SHARED_MEMORY_AVAILABLE:
        try:
            df = reconstruct_dataframe_from_shared_memory(shm_info)
        except Exception as e:
            log_warn(f"Failed to reconstruct DataFrame from shared memory: {e}")
            # Fallback to pickle if available
            if df_bytes is not None:
                df = pickle.loads(df_bytes)
            else:
                raise
    elif df_bytes is not None:
        # Unpickle DataFrame
        df = pickle.loads(df_bytes)
    else:
        raise ValueError("Either df_bytes or shm_info must be provided")

    # Create new DataFetcher instance for this worker
    # (DataFetcher contains connection objects that can't be pickled)
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)

    # Create a new HybridSignalCalculator instance for this worker
    hybrid_calc = HybridSignalCalculator(
        data_fetcher=data_fetcher,
        enabled_indicators=enabled_indicators or [],
        use_confidence_weighting=use_confidence_weighting,
        min_indicators_agreement=min_indicators_agreement,
    )

    batch_signals = {}

    # Calculate signals for each period in this batch
    for i in range(start_idx, end_idx):
        try:
            signal, confidence = hybrid_calc.calculate_single_signal_highest_confidence(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                period_index=i,
                osc_length=osc_length,
                osc_mult=osc_mult,
                osc_strategies=osc_strategies,
                spc_params=spc_params,
            )
            batch_signals[i] = signal
        except Exception as e:
            log_warn(f"Error calculating signal for period {i} in batch [{start_idx}:{end_idx}]: {e}")
            batch_signals[i] = 0  # Default to no signal on error

    return batch_signals


# Backward compatibility alias
_calculate_signal_batch_worker = calculate_signal_batch_worker
