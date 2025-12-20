"""
Signal calculation for backtester.

This module contains functions for calculating trading signals,
both sequentially and in parallel using multiprocessing.
"""

from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import pickle
import signal
import functools
from multiprocessing import Pool, cpu_count

from modules.common.utils import (
    log_error,
    log_warn,
    log_progress,
)
from modules.common.ui.progress_bar import ProgressBar
from config.position_sizing import (
    HYBRID_OSC_LENGTH,
    HYBRID_OSC_MULT,
    HYBRID_OSC_STRATEGIES,
    HYBRID_SPC_PARAMS,
    ENABLE_PARALLEL_PROCESSING,
    NUM_WORKERS,
    BATCH_SIZE,
    OPTIMIZE_BATCH_SIZE,
    MIN_BATCH_SIZE,
    MAX_BATCH_SIZE,
    BATCH_SIZE_OVERHEAD_FACTOR,
    LOG_PERFORMANCE_METRICS,
)
from .parallel_workers import calculate_signal_batch_worker, calculate_single_signal_batch_worker
from .exit_conditions import check_long_exit_conditions, check_short_exit_conditions
from .signal_calculator_incremental import (
    calculate_signals_incremental,
    calculate_single_signals_incremental,
)

# Try to import shared memory utilities
try:
    from .shared_memory_utils import (
        setup_shared_memory_for_dataframe,
        cleanup_shared_memory,
        SHARED_MEMORY_AVAILABLE,
    )
except ImportError:
    SHARED_MEMORY_AVAILABLE = False
    def setup_shared_memory_for_dataframe(df: pd.DataFrame) -> dict:
        raise RuntimeError("Shared memory utilities not available")
    def cleanup_shared_memory(shm_info: dict) -> None:
        pass


def calculate_signals(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    limit: int,
    signal_type: str,
    hybrid_signal_calculator,
) -> pd.Series:
    """
    Calculate signals for each period in the DataFrame using hybrid approach.
    
    Uses vectorized pre-computation: calculates all indicators once for the entire
    DataFrame, then extracts signals for each period incrementally. This maintains
    walk-forward semantics while being much faster than recalculating indicators
    for each period.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading pair symbol
        timeframe: Timeframe for data
        limit: Number of candles to look back
        signal_type: "LONG" or "SHORT"
        hybrid_signal_calculator: HybridSignalCalculator instance
        
    Returns:
        Series with signal values (1 for LONG entry, -1 for SHORT entry, 0 for no signal)
    """
    signals = pd.Series(0, index=df.index)
    
    log_progress(f"  Pre-computing indicators for {len(df)} periods (vectorized)...")
    print()  # Newline before progress bar
    
    # STEP 1: Pre-compute all indicators (once) - this is much faster than per-period calculation
    try:
        precomputed_indicators = hybrid_signal_calculator.precompute_all_indicators_vectorized(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            osc_length=HYBRID_OSC_LENGTH,
            osc_mult=HYBRID_OSC_MULT,
            osc_strategies=HYBRID_OSC_STRATEGIES,
            spc_params=HYBRID_SPC_PARAMS,
        )
    except Exception as e:
        log_error(f"Error in vectorized indicator pre-computation: {e}")
        log_warn("Falling back to sequential calculation...")
        # Fallback to original sequential method
        log_progress(f"  Calculating hybrid signals for {len(df)} periods (sequential fallback)...")
        print()  # Newline before progress bar
        progress = ProgressBar(total=len(df), label="Calculating signals")
        try:
            for i in range(len(df)):
                try:
                    signal, confidence = hybrid_signal_calculator.calculate_hybrid_signal(
                        df=df,
                        symbol=symbol,
                        timeframe=timeframe,
                        period_index=i,
                        signal_type=signal_type,
                        osc_length=HYBRID_OSC_LENGTH,
                        osc_mult=HYBRID_OSC_MULT,
                        osc_strategies=HYBRID_OSC_STRATEGIES,
                        spc_params=HYBRID_SPC_PARAMS,
                    )
                    signals.iloc[i] = signal
                except Exception as e2:
                    log_warn(f"Error calculating signal for period {i}: {e2}")
                    signals.iloc[i] = 0
                progress.update()
        finally:
            progress.finish()
        
        # Log signal statistics
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        neutral_signals = (signals == 0).sum()
        log_progress(f"  Signal distribution: LONG={long_signals}, SHORT={short_signals}, NEUTRAL={neutral_signals}")
        return signals
    
    # STEP 2: Calculate signals incrementally from pre-computed indicators
    log_progress(f"  Calculating signals from pre-computed indicators...")
    print()  # Newline before progress bar
    
    # Initialize progress bar
    progress = ProgressBar(total=len(df), label="Calculating signals")
    
    try:
        for i in range(len(df)):
            try:
                # Calculate signal from pre-computed indicators (maintains walk-forward semantics)
                signal, confidence = hybrid_signal_calculator.calculate_signal_from_precomputed(
                    precomputed_indicators=precomputed_indicators,
                    period_index=i,
                    signal_type=signal_type,
                )
                signals.iloc[i] = signal
            except Exception as e:
                log_warn(f"Error calculating signal for period {i}: {e}")
                signals.iloc[i] = 0  # Default to no signal on error
            
            # Update progress bar
            progress.update()
    finally:
        # Ensure progress bar is finished even if there's an error
        progress.finish()
    
    # Log signal statistics
    long_signals = (signals == 1).sum()
    short_signals = (signals == -1).sum()
    neutral_signals = (signals == 0).sum()
    log_progress(f"  Signal distribution: LONG={long_signals}, SHORT={short_signals}, NEUTRAL={neutral_signals}")
    
    # Log cache statistics
    cache_stats = hybrid_signal_calculator.get_cache_stats()
    log_progress(f"  Cache stats: {cache_stats['signal_cache_size']}/{cache_stats['signal_cache_max_size']} signals cached")
    
    return signals


def calculate_single_signals(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    limit: int,
    hybrid_signal_calculator,
) -> pd.Series:
    """
    Calculate signals for each period using single signal (highest confidence) approach.
    
    Uses vectorized pre-computation: calculates all indicators once for the entire
    DataFrame, then extracts signals for each period incrementally. This maintains
    walk-forward semantics while being much faster than recalculating indicators
    for each period.
    
    Unlike calculate_signals(), this method:
    - Does NOT require majority vote
    - Does NOT filter by expected signal_type
    - Simply selects the signal with highest confidence from all indicators
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading pair symbol
        timeframe: Timeframe for data
        limit: Number of candles to look back
        hybrid_signal_calculator: HybridSignalCalculator instance
        
    Returns:
        Series with signal values (1 for LONG entry, -1 for SHORT entry, 0 for no signal)
    """
    signals = pd.Series(0, index=df.index)
    
    log_progress(f"  Pre-computing indicators for {len(df)} periods (vectorized)...")
    print()  # Newline before progress bar
    
    # STEP 1: Pre-compute all indicators (once) - this is much faster than per-period calculation
    try:
        precomputed_indicators = hybrid_signal_calculator.precompute_all_indicators_vectorized(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            osc_length=HYBRID_OSC_LENGTH,
            osc_mult=HYBRID_OSC_MULT,
            osc_strategies=HYBRID_OSC_STRATEGIES,
            spc_params=HYBRID_SPC_PARAMS,
        )
    except Exception as e:
        log_error(f"Error in vectorized indicator pre-computation: {e}")
        log_warn("Falling back to sequential calculation...")
        # Fallback to original sequential method
        log_progress(f"  Calculating single signals for {len(df)} periods (sequential fallback)...")
        print()  # Newline before progress bar
        progress = ProgressBar(total=len(df), label="Calculating signals")
        try:
            for i in range(len(df)):
                try:
                    signal, confidence = hybrid_signal_calculator.calculate_single_signal_highest_confidence(
                        df=df,
                        symbol=symbol,
                        timeframe=timeframe,
                        period_index=i,
                        osc_length=HYBRID_OSC_LENGTH,
                        osc_mult=HYBRID_OSC_MULT,
                        osc_strategies=HYBRID_OSC_STRATEGIES,
                        spc_params=HYBRID_SPC_PARAMS,
                    )
                    signals.iloc[i] = signal
                except Exception as e2:
                    log_warn(f"Error calculating signal for period {i}: {e2}")
                    signals.iloc[i] = 0
                progress.update()
        finally:
            progress.finish()
        
        # Log signal statistics
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        neutral_signals = (signals == 0).sum()
        log_progress(f"  Signal distribution: LONG={long_signals}, SHORT={short_signals}, NEUTRAL={neutral_signals}")
        return signals
    
    # STEP 2: Calculate signals incrementally from pre-computed indicators
    log_progress(f"  Calculating signals from pre-computed indicators (single signal mode)...")
    print()  # Newline before progress bar
    
    # Initialize progress bar
    progress = ProgressBar(total=len(df), label="Calculating signals")
    
    try:
        for i in range(len(df)):
            try:
                # Calculate signal from pre-computed indicators (maintains walk-forward semantics)
                signal, confidence = hybrid_signal_calculator.calculate_single_signal_from_precomputed(
                    precomputed_indicators=precomputed_indicators,
                    period_index=i,
                )
                signals.iloc[i] = signal
            except Exception as e:
                log_warn(f"Error calculating signal for period {i}: {e}")
                signals.iloc[i] = 0  # Default to no signal on error
            
            # Update progress bar
            progress.update()
    finally:
        # Ensure progress bar is finished even if there's an error
        progress.finish()
    
    # Log signal statistics
    long_signals = (signals == 1).sum()
    short_signals = (signals == -1).sum()
    neutral_signals = (signals == 0).sum()
    log_progress(f"  Signal distribution: LONG={long_signals}, SHORT={short_signals}, NEUTRAL={neutral_signals}")
    
    # Log cache statistics
    cache_stats = hybrid_signal_calculator.get_cache_stats()
    log_progress(f"  Cache stats: {cache_stats['signal_cache_size']}/{cache_stats['signal_cache_max_size']} signals cached")
    
    return signals


def calculate_single_signals_parallel(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    limit: int,
    hybrid_signal_calculator,
    fallback_calculate_single_signals,
) -> pd.Series:
    """
    Calculate signals for each period using multiprocessing (batch processing) with single signal mode.
    
    Divides periods into batches and processes them in parallel.
    Uses single signal (highest confidence) approach instead of majority vote.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading pair symbol
        timeframe: Timeframe for data
        limit: Number of candles to look back
        hybrid_signal_calculator: HybridSignalCalculator instance (for config access)
        fallback_calculate_single_signals: Function to fall back to sequential calculation on error
        
    Returns:
        Series with signal values (1 for LONG entry, -1 for SHORT entry, 0 for no signal)
    """
    signals = pd.Series(0, index=df.index)
    
    # Determine number of workers
    num_workers = NUM_WORKERS if NUM_WORKERS is not None else cpu_count()
    num_workers = min(num_workers, len(df))  # Don't use more workers than periods
    
    # OPTIMIZATION: Calculate optimal batch size with improved algorithm
    if BATCH_SIZE is not None:
        batch_size = BATCH_SIZE
    elif OPTIMIZE_BATCH_SIZE:
        # Improved dynamic batch sizing algorithm
        base_batch_size = max(MIN_BATCH_SIZE, len(df) // (num_workers * BATCH_SIZE_OVERHEAD_FACTOR))
        
        # Adjust based on DataFrame size
        if len(df) > 5000:
            size_factor = 1.5
        elif len(df) > 2000:
            size_factor = 1.2
        else:
            size_factor = 1.0
        
        # Adjust based on worker count
        worker_factor = 1.0 / (1.0 + (num_workers - 1) * 0.1)
        
        optimal_batch_size = int(base_batch_size * size_factor * worker_factor)
        
        # Clamp to reasonable bounds
        batch_size = max(MIN_BATCH_SIZE, min(optimal_batch_size, MAX_BATCH_SIZE, len(df) // max(1, num_workers)))
    else:
        batch_size = max(1, len(df) // num_workers)
    
    # Create batches
    batches = []
    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        batches.append((i, end_idx))
    
    log_progress(f"  Calculating signals in parallel (single signal mode): {len(batches)} batches, {num_workers} workers")
    print()  # Newline before progress bar
    
    # Initialize progress bar
    progress = ProgressBar(total=len(batches), label="Processing batches")
    
    # Prepare arguments for worker function
    calc_args = {
        'symbol': symbol,
        'timeframe': timeframe,
        'limit': limit,
        'osc_length': HYBRID_OSC_LENGTH,
        'osc_mult': HYBRID_OSC_MULT,
        'osc_strategies': HYBRID_OSC_STRATEGIES,
        'spc_params': HYBRID_SPC_PARAMS,
        'enabled_indicators': hybrid_signal_calculator.enabled_indicators,
        'use_confidence_weighting': hybrid_signal_calculator.use_confidence_weighting,
        'min_indicators_agreement': hybrid_signal_calculator.min_indicators_agreement,
    }
    
    # OPTIMIZATION: Try shared memory first (more efficient), fallback to pickle
    shm_info = None
    df_bytes = None
    
    if SHARED_MEMORY_AVAILABLE:
        try:
            shm_info = setup_shared_memory_for_dataframe(df)
            if LOG_PERFORMANCE_METRICS:
                # Estimate shared memory size
                total_size = sum(
                    np.prod(info['shape']) * np.dtype(info['dtype']).itemsize
                    for info in shm_info['shm_objects'].values()
                )
                total_size_mb = total_size / (1024 * 1024)
                log_progress(f"  Using shared memory: {total_size_mb:.2f} MB")
        except Exception as e:
            log_warn(f"Failed to setup shared memory: {e}")
            log_warn("Falling back to pickle...")
            shm_info = None
    
    # Fallback to pickle if shared memory not available or failed
    if shm_info is None:
        try:
            df_bytes = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
            # Log DataFrame size for monitoring
            if LOG_PERFORMANCE_METRICS:
                df_size_mb = len(df_bytes) / (1024 * 1024)
                log_progress(f"  DataFrame size: {df_size_mb:.2f} MB (pickled)")
        except (pickle.PicklingError, MemoryError) as e:
            log_warn(f"Failed to pickle DataFrame with HIGHEST_PROTOCOL, falling back to default: {e}")
            try:
                df_bytes = pickle.dumps(df)  # Fallback to default protocol
            except (pickle.PicklingError, MemoryError) as e2:
                log_error(f"Failed to pickle DataFrame even with default protocol: {e2}")
                log_warn("Falling back to sequential calculation due to memory constraints")
                signals = fallback_calculate_single_signals(df, symbol, timeframe, limit, hybrid_signal_calculator)
                progress.finish()
                return signals
    
    # Use multiprocessing Pool with signal handling for graceful shutdown
    pool = None
    original_handler = None
    
    # Use a list to store pool reference for signal handler
    pool_ref = [None]
    
    def signal_handler(signum, frame):
        """Handle Ctrl+C to gracefully terminate all worker processes."""
        log_warn("\nReceived interrupt signal. Terminating worker processes...")
        if pool_ref[0] is not None:
            try:
                pool_ref[0].terminate()
                pool_ref[0].join(timeout=2)
            except Exception:
                pass
        # Restore original handler and raise KeyboardInterrupt
        if original_handler is not None:
            signal.signal(signal.SIGINT, original_handler)
        raise KeyboardInterrupt("Interrupted by user")
    
    try:
        # Register signal handler before creating pool
        original_handler = signal.signal(signal.SIGINT, signal_handler)
        
        pool = Pool(processes=num_workers)
        pool_ref[0] = pool  # Store reference for signal handler
        
        # Create partial function with fixed arguments
        # Pass either shared memory info or pickled bytes
        worker_func = functools.partial(
            calculate_single_signal_batch_worker,
            df_bytes=df_bytes if shm_info is None else None,
            shm_info=shm_info if shm_info is not None else None,
            **calc_args
        )
        
        # Process batches in parallel
        results = pool.starmap(worker_func, batches)
        
        # Restore original signal handler on success
        if original_handler is not None:
            signal.signal(signal.SIGINT, original_handler)
        
        # Merge results and update progress
        for batch_signals in results:
            if batch_signals is not None:
                for idx, signal_val in batch_signals.items():
                    signals.iloc[idx] = signal_val
            # Update progress after each batch is merged
            progress.update()
    except KeyboardInterrupt:
        # Clean up pool on interrupt
        if pool is not None:
            pool.terminate()
            pool.join()
        # Clean up shared memory
        if shm_info is not None:
            try:
                cleanup_shared_memory(shm_info)
            except Exception:
                pass
        progress.finish()
        raise  # Re-raise to propagate interrupt
    except Exception as e:
        log_error(f"Error in parallel signal calculation: {e}")
        log_warn("Falling back to sequential calculation")
        signals = fallback_calculate_single_signals(df, symbol, timeframe, limit, hybrid_signal_calculator)
    finally:
        # Clean up shared memory if used
        if shm_info is not None:
            try:
                cleanup_shared_memory(shm_info)
            except Exception as e:
                log_warn(f"Error cleaning up shared memory: {e}")
        
        # Ensure progress bar is finished and pool is closed
        if pool is not None:
            pool.close()
            pool.join()
        progress.finish()
    
    # Log signal statistics
    long_signals = (signals == 1).sum()
    short_signals = (signals == -1).sum()
    neutral_signals = (signals == 0).sum()
    log_progress(f"  Signal distribution: LONG={long_signals}, SHORT={short_signals}, NEUTRAL={neutral_signals}")
    
    return signals


def calculate_signals_parallel(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    limit: int,
    signal_type: str,
    hybrid_signal_calculator,
    fallback_calculate_signals,
) -> pd.Series:
    """
    Calculate signals for each period using multiprocessing (batch processing).
    
    Divides periods into batches and processes them in parallel.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading pair symbol
        timeframe: Timeframe for data
        limit: Number of candles to look back
        signal_type: "LONG" or "SHORT"
        hybrid_signal_calculator: HybridSignalCalculator instance (for config access)
        fallback_calculate_signals: Function to fall back to sequential calculation on error
        
    Returns:
        Series with signal values (1 for LONG entry, -1 for SHORT entry, 0 for no signal)
    """
    signals = pd.Series(0, index=df.index)
    
    # Determine number of workers
    num_workers = NUM_WORKERS if NUM_WORKERS is not None else cpu_count()
    num_workers = min(num_workers, len(df))  # Don't use more workers than periods
    
    # OPTIMIZATION: Calculate optimal batch size with improved algorithm
    if BATCH_SIZE is not None:
        batch_size = BATCH_SIZE
    elif OPTIMIZE_BATCH_SIZE:
        # Improved dynamic batch sizing algorithm
        # Factors to consider:
        # 1. DataFrame size (larger = larger batches)
        # 2. Number of workers (more workers = smaller batches to balance load)
        # 3. Minimum batch size to avoid overhead
        # 4. Maximum batch size to avoid memory issues
        
        # Base batch size: balance between overhead and memory
        base_batch_size = max(MIN_BATCH_SIZE, len(df) // (num_workers * BATCH_SIZE_OVERHEAD_FACTOR))
        
        # Adjust based on DataFrame size
        # For very large DataFrames, use larger batches
        if len(df) > 5000:
            size_factor = 1.5
        elif len(df) > 2000:
            size_factor = 1.2
        else:
            size_factor = 1.0
        
        # Adjust based on worker count
        # More workers = slightly smaller batches to improve load balancing
        worker_factor = 1.0 / (1.0 + (num_workers - 1) * 0.1)
        
        optimal_batch_size = int(base_batch_size * size_factor * worker_factor)
        
        # Clamp to reasonable bounds
        batch_size = max(MIN_BATCH_SIZE, min(optimal_batch_size, MAX_BATCH_SIZE, len(df) // max(1, num_workers)))
    else:
        batch_size = max(1, len(df) // num_workers)
    
    # Create batches
    batches = []
    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        batches.append((i, end_idx))
    
    log_progress(f"  Calculating signals in parallel: {len(batches)} batches, {num_workers} workers")
    print()  # Newline before progress bar
    
    # Initialize progress bar
    progress = ProgressBar(total=len(batches), label="Processing batches")
    
    # Prepare arguments for worker function
    # We need to pass the hybrid_signal_calculator config, not the object itself
    # (multiprocessing can't pickle complex objects)
    calc_args = {
        'symbol': symbol,
        'timeframe': timeframe,
        'limit': limit,
        'signal_type': signal_type,
        'osc_length': HYBRID_OSC_LENGTH,
        'osc_mult': HYBRID_OSC_MULT,
        'osc_strategies': HYBRID_OSC_STRATEGIES,
        'spc_params': HYBRID_SPC_PARAMS,
        'enabled_indicators': hybrid_signal_calculator.enabled_indicators,
        'use_confidence_weighting': hybrid_signal_calculator.use_confidence_weighting,
        'min_indicators_agreement': hybrid_signal_calculator.min_indicators_agreement,
    }
    
    # OPTIMIZATION: Try shared memory first (more efficient), fallback to pickle
    shm_info = None
    df_bytes = None
    
    if SHARED_MEMORY_AVAILABLE:
        try:
            shm_info = setup_shared_memory_for_dataframe(df)
            if LOG_PERFORMANCE_METRICS:
                # Estimate shared memory size
                total_size = sum(
                    np.prod(info['shape']) * np.dtype(info['dtype']).itemsize
                    for info in shm_info['shm_objects'].values()
                )
                total_size_mb = total_size / (1024 * 1024)
                log_progress(f"  Using shared memory: {total_size_mb:.2f} MB")
        except Exception as e:
            log_warn(f"Failed to setup shared memory: {e}")
            log_warn("Falling back to pickle...")
            shm_info = None
    
    # Fallback to pickle if shared memory not available or failed
    if shm_info is None:
        try:
            df_bytes = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
            # Log DataFrame size for monitoring
            if LOG_PERFORMANCE_METRICS:
                df_size_mb = len(df_bytes) / (1024 * 1024)
                log_progress(f"  DataFrame size: {df_size_mb:.2f} MB (pickled)")
        except (pickle.PicklingError, MemoryError) as e:
            log_warn(f"Failed to pickle DataFrame with HIGHEST_PROTOCOL, falling back to default: {e}")
            try:
                df_bytes = pickle.dumps(df)  # Fallback to default protocol
            except (pickle.PicklingError, MemoryError) as e2:
                log_error(f"Failed to pickle DataFrame even with default protocol: {e2}")
                log_warn("Falling back to sequential calculation due to memory constraints")
                signals = fallback_calculate_signals(df, symbol, timeframe, limit, signal_type)
                progress.finish()
                return signals
    
    # Use multiprocessing Pool with signal handling for graceful shutdown
    pool = None
    original_handler = None
    
    # Use a list to store pool reference for signal handler
    pool_ref = [None]
    
    def signal_handler(signum, frame):
        """Handle Ctrl+C to gracefully terminate all worker processes."""
        log_warn("\nReceived interrupt signal. Terminating worker processes...")
        if pool_ref[0] is not None:
            try:
                pool_ref[0].terminate()
                pool_ref[0].join(timeout=2)
            except Exception:
                pass
        # Restore original handler and raise KeyboardInterrupt
        if original_handler is not None:
            signal.signal(signal.SIGINT, original_handler)
        raise KeyboardInterrupt("Interrupted by user")
    
    try:
        # Register signal handler before creating pool
        original_handler = signal.signal(signal.SIGINT, signal_handler)
        
        pool = Pool(processes=num_workers)
        pool_ref[0] = pool  # Store reference for signal handler
        
        # Create partial function with fixed arguments
        # Pass either shared memory info or pickled bytes
        worker_func = functools.partial(
            calculate_signal_batch_worker,
            df_bytes=df_bytes if shm_info is None else None,
            shm_info=shm_info if shm_info is not None else None,
            **calc_args
        )
        
        # Process batches in parallel
        results = pool.starmap(worker_func, batches)
        
        # Restore original signal handler on success
        if original_handler is not None:
            signal.signal(signal.SIGINT, original_handler)
        
        # Merge results and update progress
        for batch_signals in results:
            if batch_signals is not None:
                for idx, signal_val in batch_signals.items():
                    signals.iloc[idx] = signal_val
            # Update progress after each batch is merged
            progress.update()
    except KeyboardInterrupt:
        # Clean up pool on interrupt
        if pool is not None:
            pool.terminate()
            pool.join()
        # Clean up shared memory
        if shm_info is not None:
            try:
                cleanup_shared_memory(shm_info)
            except Exception:
                pass
        progress.finish()
        raise  # Re-raise to propagate interrupt
    except Exception as e:
        log_error(f"Error in parallel signal calculation: {e}")
        log_warn("Falling back to sequential calculation")
        signals = fallback_calculate_signals(df, symbol, timeframe, limit, signal_type)
    finally:
        # Clean up shared memory if used
        if shm_info is not None:
            try:
                cleanup_shared_memory(shm_info)
            except Exception as e:
                log_warn(f"Error cleaning up shared memory: {e}")
        
        # Ensure progress bar is finished and pool is closed
        if pool is not None:
            pool.close()
            pool.join()
        progress.finish()
    
    # Log signal statistics
    long_signals = (signals == 1).sum()
    short_signals = (signals == -1).sum()
    neutral_signals = (signals == 0).sum()
    log_progress(f"  Signal distribution: LONG={long_signals}, SHORT={short_signals}, NEUTRAL={neutral_signals}")
    
    return signals

